from .BYOL2 import BYOL
from .SIMSIAM import SIMSIAM
from .SIMCLR import SIMCLR
from .DINO import DINO
from .COS_ADAPTER import COS_ADAPTER
from .common import NetWrapper
from transforms.custom_transforms import PadToSquare
import torch
import torchvision
import torchvision.transforms as T
import tensorflow as tf
import tensorflow as tf
import tensorflow_datasets as tfds
import hashlib
import os

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
    if backbone_name == 'DINOV2':
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
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
    if model_config['FRAMEWORK'] == 'COS_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'))
    if model_config['FRAMEWORK'] == 'INVERTED_COS_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='inverted')
    if model_config['FRAMEWORK'] == 'DOUBLE_COS_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='double')
    if model_config['FRAMEWORK'] == 'RESIDUAL_COS_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='residual')
    if model_config['FRAMEWORK'] == 'RESIDUAL_INVERTED_COS_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='residual_inverted')
    if model_config['FRAMEWORK'] == 'RESIDUAL_DOUBLE_COS_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='residual_double')
    if model_config['FRAMEWORK'] == 'CE_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='ce_adapter')
    if model_config['FRAMEWORK'] == 'INVERTED_CE_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='inverted_ce_adapter')
    if model_config['FRAMEWORK'] == 'DOUBLE_CE_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='double_ce_adapter')
    if model_config['FRAMEWORK'] == 'RESIDUAL_CE_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='residual_ce_adapter')
    if model_config['FRAMEWORK'] == 'RESIDUAL_INVERTED_CE_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='residual_inverted_ce_adapter')
    if model_config['FRAMEWORK'] == 'RESIDUAL_DOUBLE_CE_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='residual_double_ce_adapter')
    if model_config['FRAMEWORK'] == 'WITHOUT_ADAPTER':
        model = COS_ADAPTER(backbone, model_config.getint('CROP_SIZE'), mode='without_adapter')
    return model

def get_dataset(model_config, train=True):
    with tf.device('/CPU:0'):
        ds_name = model_config['DATASET']
        CROP_SIZE = model_config.getint('CROP_SIZE')
        if train == True:
            if ds_name == 'SKETCHY':
                ds = tfds.load('tfds_sketchy', split='train', as_supervised=True, data_dir='/home/wcampos/tensorflow_datasets/')
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
                ds = tfds.load('tfds_ecommerce_train', split='pidinet', as_supervised=True, data_dir='/home/wcampos/tensorflow_datasets/')
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
            elif ds_name == 'FLICKR':
                ds = tfds.load('tfds_flickr25k', split='train', as_supervised=True, data_dir='/home/wcampos/tensorflow_datasets/')
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
            elif ds_name == 'QDEXT':
                ds = tfds.load('tfds_quickdraw_extended_train', split='train', as_supervised=True, data_dir='/home/wcampos/tensorflow_datasets/')
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
                ds = tfds.load('tfds_sketchy', split='validation_known', as_supervised=True, data_dir='/home/wcampos/tensorflow_datasets/')
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
            elif ds_name == 'SKETCHY_UNKNOWN':
                ds = tfds.load('tfds_sketchy', split='validation_unknown', as_supervised=True, data_dir='/home/wcampos/tensorflow_datasets/')
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
                queries = tfds.load('tfds_ecommerce_valid', split='sketches', as_supervised=True, data_dir='/home/wcampos/tensorflow_datasets/')
                catalogue = tfds.load('tfds_ecommerce_valid', split='photos', as_supervised=True, data_dir='/home/wcampos/tensorflow_datasets/')
                queries = list(queries.as_numpy_iterator())
                catalogue = list(catalogue.as_numpy_iterator())
                image_transform = T.Compose([
                    lambda x: torch.from_numpy(x),
                    lambda x: x.permute(2, 0, 1),
                    PadToSquare(255),
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
            elif ds_name == 'FLICKR':
                queries = tfds.load('tfds_flickr15k', split='sketches', as_supervised=True, data_dir='/home/wcampos/tensorflow_datasets/')
                catalogue = tfds.load('tfds_flickr15k', split='photos', as_supervised=True, data_dir='/home/wcampos/tensorflow_datasets/')
                queries = list(queries.as_numpy_iterator())
                catalogue = list(catalogue.as_numpy_iterator())
                image_transform = T.Compose([
                    lambda x: torch.from_numpy(x),
                    lambda x: x.permute(2, 0, 1),
                    PadToSquare(255),
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
            elif ds_name == 'QDEXT':
                queries = tfds.load(
                    'tfds_quickdraw_extended_valid',
                    split='sketches_known',
                    as_supervised=True,
                    data_dir='/home/wcampos/tensorflow_datasets/')
                queries = queries.batch(model_config.getint('BATCH_SIZE'))
                queries = tfds.as_numpy(queries)
                catalogue = tfds.load(
                    'tfds_quickdraw_extended_valid',
                    split='photos_known',
                    as_supervised=True,
                    data_dir='/home/wcampos/tensorflow_datasets/'
                    )
                catalogue = catalogue.batch(model_config.getint('BATCH_SIZE'))
                catalogue = tfds.as_numpy(catalogue)
            elif ds_name == 'QDEXT_UNKNOWN':
                queries = tfds.load(
                    'tfds_quickdraw_extended_valid',
                    split='sketches_unknown',
                    as_supervised=True,
                    data_dir='/home/wcampos/tensorflow_datasets/'
                    )
                queries = queries.batch(model_config.getint('BATCH_SIZE'))
                queries = tfds.as_numpy(queries)
                catalogue = tfds.load(
                    'tfds_quickdraw_extended_valid',
                    split='photos_unknown_full',
                    as_supervised=True,
                    data_dir='/home/wcampos/tensorflow_datasets/'
                    )
                catalogue = catalogue.batch(model_config.getint('BATCH_SIZE'))
                catalogue = tfds.as_numpy(catalogue)
            return queries, catalogue
