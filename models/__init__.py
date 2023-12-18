from .BYOL2 import BYOL
from .SIMSIAM import SIMSIAM
from .SIMCLR import SIMCLR
from .DINO import DINO
from .common import NetWrapper
from transforms.custom_transforms import PadToSquare
import torch
import torchvision
import torchvision.transforms as T
import tensorflow as tf
import tensorflow as tf
import tensorflow_datasets as tfds
import hashlib

def delete_duplicates_and_split(pairs_dataset, sketch_transform, photo_transform):
    unique_sketches = set()
    unique_photos = set()
    sketches = []
    photos = []
    for item in pairs_dataset:
        s, p, l = item
        s_hash = hashlib.sha1(s).hexdigest()
        p_hash = hashlib.sha1(p).hexdigest()
        if s_hash not in unique_sketches:
            unique_sketches.add(s_hash)
            sketches.append((sketch_transform(s), l))
        if p_hash not in unique_photos:
            unique_photos.add(p_hash)
            photos.append((photo_transform(p), l))
    return sketches, photos

def get_backbone(backbone_name):
    if backbone_name == 'RESNET50':
        backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        backbone = NetWrapper(backbone, layer='avgpool')
    if backbone_name == 'VITB16':
        backbone = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        # backbone.heads = torch.nn.Identity()
    return backbone

def get_model(model_config):
    backbone = get_backbone(model_config['BACKBONE'])
    if model_config['FRAMEWORK'] == 'BYOL':
        model = BYOL(backbone, model_config.getint('CROP_SIZE'), cosine_ema_steps=model_config.getint('TOTAL_EPOCHS')*model_config.getint('DS_LEN'))
    if model_config['FRAMEWORK'] == 'SIMSIAM':
        model = SIMSIAM(backbone, model_config.getint('CROP_SIZE'))
    if model_config['FRAMEWORK'] == 'SIMCLR':
        model = SIMCLR(backbone, model_config.getint('CROP_SIZE'))
    if model_config['FRAMEWORK'] == 'DINO':
        model = DINO(backbone, model_config.getint('CROP_SIZE'))
    return model

def get_dataset(model_config, train=True):
    with tf.device('/CPU:0'):
        # with tf.device('/CPU:0'):
        ds_name = model_config['DATASET']
        CROP_SIZE = model_config.getint('CROP_SIZE')
        if train == True:
            if ds_name == 'SKETCHY':
                ds = tfds.load('tfds_sketchy', split='train', as_supervised=True, data_dir='/home/wcampos/data/tensorflow_datasets/')
                ds = list(ds.as_numpy_iterator())
                ds_len = len(ds)
                train_loader = torch.utils.data.DataLoader(
                    ds,
                    batch_size=model_config.getint('BATCH_SIZE'),
                    shuffle=True,
                    collate_fn=lambda x: [(torch.from_numpy(a), torch.from_numpy(b)) for a, b, _ in x],
                    num_workers=model_config.getint('DATALOADER_WORKERS'),
                    drop_last=True,
                )
            elif ds_name == 'ECOMMERCE':
                ds = tfds.load('tfds_ecommerce_train', split='pidinet', as_supervised=True, data_dir='/home/wcampos/data/tensorflow_datasets/')
                ds = list(ds.as_numpy_iterator())
                ds_len = len(ds)
                train_loader = torch.utils.data.DataLoader(
                    ds,
                    batch_size=model_config.getint('BATCH_SIZE'),
                    shuffle=True,
                    collate_fn=lambda x: [(torch.from_numpy(a), torch.from_numpy(b)) for a, b, _ in x],
                    num_workers=model_config.getint('DATALOADER_WORKERS'),
                    drop_last=True,
                )
            return train_loader, ds_len
        else:
            if ds_name == 'SKETCHY':
                ds = tfds.load('tfds_sketchy', split='validation', as_supervised=True, data_dir='/home/wcampos/data/tensorflow_datasets/')
                ds = list(ds.as_numpy_iterator())
                image_transform = T.Compose([
                    lambda x: torch.from_numpy(x),
                    lambda x: x.permute(2, 0, 1),
                    T.Resize((CROP_SIZE, CROP_SIZE)),
                    lambda x: x / 255
                ])
                sketch_transform = T.Compose([
                    lambda x: torch.from_numpy(x),
                    lambda x: x.permute(2, 0, 1),
                    T.Resize((CROP_SIZE, CROP_SIZE)),
                    lambda x: x / 255
                ])
                queries, catalogue = delete_duplicates_and_split(ds, sketch_transform, image_transform)
            elif ds_name == 'ECOMMERCE':
                queries = tfds.load('tfds_ecommerce_valid', split='sketches', as_supervised=True, data_dir='/home/wcampos/data/tensorflow_datasets/')
                catalogue = tfds.load('tfds_ecommerce_valid', split='photos', as_supervised=True, data_dir='/home/wcampos/data/tensorflow_datasets/')
                queries = list(queries.as_numpy_iterator())
                catalogue = list(catalogue.as_numpy_iterator())
                image_transform = T.Compose([
                    lambda x: torch.from_numpy(x),
                    lambda x: x.permute(2, 0, 1),
                    T.Resize((CROP_SIZE, CROP_SIZE)),
                    lambda x: x / 255
                ])
                sketch_transform = T.Compose([
                    lambda x: torch.from_numpy(x),
                    lambda x: x.permute(2, 0, 1),
                    PadToSquare(255),
                    T.Resize((CROP_SIZE, CROP_SIZE)),
                    lambda x: x / 255
                ])
                queries = [(sketch_transform(a), b) for a,b in queries]
                catalogue = [(image_transform(a), b) for a,b in catalogue]
            return queries, catalogue
