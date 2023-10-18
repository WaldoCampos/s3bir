from .BYOL2 import BYOL
from .SIMSIAM import SIMSIAM
from .SIMCLR import SIMCLR
from .common import NetWrapper
import torchvision

def get_backbone(backbone_name):
    if backbone_name == 'RESNET50':
        backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        backbone = NetWrapper(backbone, layer='avgpool')
    if backbone_name == 'VITB16':
        backbone = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        backbone = NetWrapper(backbone, layer='encoder')
    return backbone

def get_model(model_config):
    backbone = get_backbone(model_config['BACKBONE'])
    if model_config['FRAMEWORK'] == 'BYOL':
        model = BYOL(backbone, model_config.getint('CROP_SIZE'))
    if model_config['FRAMEWORK'] == 'SIMSIAM':
        model = SIMSIAM(backbone, model_config.getint('CROP_SIZE'))
    if model_config['FRAMEWORK'] == 'SIMCLR':
        model = SIMCLR(backbone, model_config.getint('CROP_SIZE'))
    return model
