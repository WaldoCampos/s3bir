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

def CE_loss(z1, z2, temperature=0.5):
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
        if self.mode in ['double', 'residual_double', 'double_ce_adapter', 'residual_double_ce_adapter']:
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
        if self.mode in ['inverted', 'inverted_ce_adapter']:
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
        if self.mode in ['residual_inverted', 'residual_inverted_ce_adapter']:
            # for inverted cosine adapter reverse the inputs and the embedding return
            if return_embedding == 'online':
                return self.encoder(x)
            elif return_embedding == 'target':
                x = self.encoder(x)
                return x + self.adapter(x)
            x2, x1 = self.augment1(x), self.augment2(x)
            with torch.no_grad():
                x1 = self.encoder(x1)
                x2 = self.encoder(x2)
            x1 = x1 + self.adapter(x1)
        if self.mode in ['residual', 'residual_ce_adapter']:
            if return_embedding == 'online':
                x = self.encoder(x)
                return x + self.adapter(x)
            elif return_embedding == 'target':
                return self.encoder(x)
            x1, x2 = self.augment1(x), self.augment2(x)
            with torch.no_grad():
                x1 = self.encoder(x1)
                x2 = self.encoder(x2)
            x1 = x1 + self.adapter(x1)
        elif self.mode in ['double', 'double_ce_adapter']:
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
        elif self.mode in ['residual_double', 'residual_double_ce_adapter']:
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
        elif self.mode == 'without_adapter':
            if return_embedding == 'online':
                return self.encoder(x)
            elif return_embedding == 'target':
                return self.encoder(x)
            x1, x2 = self.augment1(x), self.augment2(x)
            with torch.no_grad():
                x1 = self.encoder(x1)
                x2 = self.encoder(x2)
            x1 = self.adapter(x1)
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
        if self.mode in ['ce_adapter', 'inverted_ce_adapter', 'double_ce_adapter', 'residual_ce_adapter', 'residual_inverted_ce_adapter', 'residual_double_ce_adapter']:
            loss = CE_loss(x1, x2)
        else:
            loss = loss_fn(x1, x2)
            loss = loss.mean()
        return loss