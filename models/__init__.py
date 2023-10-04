from .BYOL2 import BYOL
from .net_wrapper import NetWrapper
import torchvision

def get_backbone(backbone_name):
    if backbone_name == 'RESNET50':
        backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        backbone = NetWrapper(backbone, layer='avgpool')
    return backbone

def get_model(model_config):
    backbone = get_backbone(model_config['BACKBONE'])
    if model_config['FRAMEWORK'] == 'BYOL':
        model = BYOL(backbone, model_config.getint('CROP_SIZE'))
    return model