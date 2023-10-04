import copy
import math
import random
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class CosineDecayEMA():
    def __init__(self, tau, max_steps):
        super().__init__()
        self.base_tau = tau
        self.curr_step = 0
        self.max_steps = max_steps

    def update_average(self, old, new):
        if old is None:
            return new
        tau = 1 - (1-self.base_tau)*(math.cos(math.pi*self.curr_step/self.max_steps)+1)/2
        self.curr_step += 1
        return old * tau + (1 - tau) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )
    def forward(self, x):
        return self.net(x)

# main class

class BYOL(nn.Module):
    def __init__(
            self,
            encoder,
            image_size,
            projection_size = 256,
            projection_hidden_size = 4096,
            augment_fn = None,
            augment_fn2 = None,
            moving_average_decay = 0.99,
            use_momentum = True,
            cosine_ema_steps = None
    ):
        super().__init__()
        device = get_module_device(encoder)
        # self.online_encoder = NetWrapper(net, layer=hidden_layer)
        self.online_encoder = encoder

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_projector = None
        if cosine_ema_steps:
            self.target_ema_updater = CosineDecayEMA(moving_average_decay, cosine_ema_steps)
        else:
            self.target_ema_updater = EMA(moving_average_decay)

        dummy = self.online_encoder(torch.randn(2, 3, image_size, image_size, device=device))
        self.online_projector = MLP(dummy.shape[1], projection_size, projection_hidden_size)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        
        self.to(device)

        DEFAULT_AUG = T.Compose([])
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder
    
    @singleton('target_projector')
    def _get_target_projector(self):
        target_projector = copy.deepcopy(self.online_projector)
        set_requires_grad(target_projector, False)
        return target_projector

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        assert self.target_projector is not None, 'target projector has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
        update_moving_average(self.target_ema_updater, self.target_projector, self.online_projector)

    def forward(
            self,
            x,
            return_embedding = False,
            return_projection = True
    ):

        if return_embedding == 'online':
            return self.online_encoder(x)
        elif return_embedding == 'target':
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            return target_encoder(x)

        image_one, image_two = self.augment1(x), self.augment2(x)

        online_embed_one = self.online_encoder(image_one)
        online_proj_one = self.online_projector(online_embed_one)
        online_pred_one = self.online_predictor(online_proj_one)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_projector = self._get_target_projector() if self.use_momentum else self.online_projector
            target_embed_two = target_encoder(image_two)
            target_proj_two = target_projector(target_embed_two)
            target_proj_two.detach_()

        loss = loss_fn(online_pred_one, target_proj_two.detach())
        return loss.mean()
