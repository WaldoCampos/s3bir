import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from .common import MLP, get_module_device, default

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# main class

class SIMSIAM(nn.Module):
    def __init__(
            self,
            encoder,
            image_size,
            projection_size = 256,
            projection_hidden_size = 4096,
            augment_fn = None,
            augment_fn2 = None,
    ):
        super().__init__()
        device = get_module_device(encoder)
        # self.encoder = NetWrapper(net, layer=hidden_layer)
        self.encoder = encoder

        dummy = self.encoder(torch.randn(2, 3, image_size, image_size, device=device))
        self.projector = MLP(dummy.shape[1], projection_size, projection_hidden_size)
        self.predictor = MLP(projection_size, projection_size, projection_hidden_size)

        self.to(device)

        DEFAULT_AUG = T.Compose([])
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        # send a mock image tensor to instantiate singleton parameters
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
        pred_one = self.predictor(proj_one)

        with torch.no_grad():
            embed_two = self.encoder(image_two)
            proj_two = self.projector(embed_two)
            proj_two.detach_()

        loss = loss_fn(pred_one, proj_two.detach())
        return loss.mean()
