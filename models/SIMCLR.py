import copy
import math
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from .common import MLP, get_module_device, default

# helper functions

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

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)

# main class

class SIMCLR(nn.Module):
    def __init__(
            self,
            encoder,
            image_size,
            projection_size = 2048,
            projection_hidden_size = 2048,
            augment_fn = None,
            augment_fn2 = None,
    ):
        super().__init__()
        device = get_module_device(encoder)
        # self.online_encoder = NetWrapper(net, layer=hidden_layer)
        self.encoder = encoder
        
        with torch.no_grad():
            dummy = self.encoder(torch.randn(2, 3, image_size, image_size, device=device))
        self.projector = MLP(dummy.shape[1], projection_size, projection_hidden_size)
        
        self.to(device)

        DEFAULT_AUG = T.Compose([])
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        # send a mock image tensor to instantiate singleton parameters
        with torch.no_grad():
            self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    def forward(
            self,
            x,
            return_embedding = False,
    ):

        if return_embedding != False:
            return self.encoder(x)

        image_one, image_two = self.augment1(x), self.augment2(x)

        embed_one = self.encoder(image_one)
        proj_one = self.projector(embed_one)
        embed_two = self.encoder(image_two)
        proj_two = self.projector(embed_two)

        loss = loss_fn(proj_one, proj_two)
        return loss
