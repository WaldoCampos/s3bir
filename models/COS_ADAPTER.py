import torch
from torch import nn
from torchvision import transforms as T
import torch.nn.functional as F

# helper functions

def get_module_device(module):
    return next(module.parameters()).device

def default(val, def_val):
    return def_val if val is None else val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# main class
class COS_ADAPTER(nn.Module):
    def __init__(
            self,
            encoder,
            image_size,
            mode = None,
            augment_fn = None,
            augment_fn2 = None,
    ):
        super().__init__()
        device = get_module_device(encoder)
        self.mode = mode
        self.encoder = encoder
        with torch.no_grad():
            dummy = self.encoder(torch.randn(2, 3, image_size, image_size, device=device))
        self.adapter = nn.Sequential(
            nn.Linear(dummy.shape[1], 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, dummy.shape[1])
        )
        if self.mode in ['double', 'residual_double']:
            self.sketch_adapter = nn.Sequential(
                nn.Linear(dummy.shape[1], 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, dummy.shape[1])
            )

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
        if self.mode == 'inverted':
            # for inverted cosine adapter reverse the inputs and the embedding return
            if return_embedding == 'online':
                return self.encoder(x)
            elif return_embedding == 'target':
                return self.adapter(self.encoder(x))
            x2, x1 = self.augment1(x), self.augment2(x)
            with torch.no_grad():
                x1 = self.encoder(x1)
                x2 = self.encoder(x2)
            x1 = self.adapter(x1)
        if self.mode == 'double':
            if return_embedding == 'online':
                return self.adapter(self.encoder(x))
            elif return_embedding == 'target':
                return self.sketch_adapter(self.encoder(x))
            x1, x2 = self.augment1(x), self.augment2(x)
            with torch.no_grad():
                x1 = self.encoder(x1)
                x2 = self.encoder(x2)
            x1 = self.adapter(x1)
            x2 = self.sketch_adapter(x2)
        if self.mode == 'residual_double':
            if return_embedding == 'online':
                x = self.encoder(x)
                return x + self.adapter(x)
            elif return_embedding == 'target':
                x = self.encoder(x)
                return x + self.sketch_adapter(x)
            x1, x2 = self.augment1(x), self.augment2(x)
            with torch.no_grad():
                x1 = self.encoder(x1)
                x2 = self.encoder(x2)
            x1 = x1 + self.adapter(x1)
            x2 = x2 + self.sketch_adapter(x2)
        else: # this is the normal adapter case, online branch processes photos with the adapter
            if return_embedding == 'online':
                return self.adapter(self.encoder(x))
            elif return_embedding == 'target':
                return self.encoder(x)
            x1, x2 = self.augment1(x), self.augment2(x)
            with torch.no_grad():
                x1 = self.encoder(x1)
                x2 = self.encoder(x2)
            x1 = self.adapter(x1)
        loss = loss_fn(x1, x2)
        return loss.mean()