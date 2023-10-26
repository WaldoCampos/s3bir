import argparse
import configparser
import time
import datetime
import warnings
import sys
import os
import torch
import torchvision.transforms as T
import numpy as np
from models import get_model
from util.pairs_dataset import PairsDataset, pair_collate_fn
from transforms.custom_transforms import BatchTransform, SelectFromTuple, PadToSquare, ListToTensor, RandomLineSkip, RandomRotation
# from torchlars import LARS
from eval import EvalMAP
import tensorflow_datasets as tfds


def print_epoch_time(t0, epoch):
    final_time = time.time() - t0
    final_timedelta = datetime.timedelta(seconds=final_time)
    final_timedelta = final_timedelta - datetime.timedelta(microseconds=final_timedelta.microseconds)
    final_time_formatted = str(final_timedelta)
    print(f"\nTiempo en epoch {epoch+1}: {final_time_formatted}")

if __name__ == '__main__':
    # Leemos el archivo de config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to the config file', required=True)
    parser.add_argument('--device', help='what device is to be used', required=True)
    args = parser.parse_args()
    configp = configparser.ConfigParser()
    configp.read(args.config)
    config = configp['MODEL']

    SAVE_PATH = config['SAVE_PATH']
    if not os.path.exists(os.path.split(SAVE_PATH)[0]):
        os.makedirs(os.path.split(SAVE_PATH)[0])

    device = args.device
    CONTINUE = config.getint('CONTINUE')
    BATCH_SIZE = config.getint('BATCH_SIZE')
    CROP_SIZE = config.getint('CROP_SIZE')
    TOTAL_EPOCHS = config.getint('TOTAL_EPOCHS')
    LAST_EPOCH = config.getint('LAST_EPOCH')
    DATASET = config['DATASET']
    torch.cuda.empty_cache()

    # CARGA DE LOS DATOS
    
    # data_dir = {"images": TRAIN_CATALOGUE_DIR,
    #         "sketches": TRAIN_QUERY_DIR}
    # dataset = PairsDataset(
    #     data_dir["images"],
    #     data_dir["sketches"]
    # )

    ds = tfds.load(DATASET,
              split='train',
              as_supervised=True)
    ds = list(ds.as_numpy_iterator())


    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: [(torch.from_numpy(a), torch.from_numpy(b)) for a, b, _ in x],
        num_workers=config.getint('DATALOADER_WORKERS')
    )

    # FUNCIONES DE AUGMENTATION
    image_transform = T.Compose([
        BatchTransform(SelectFromTuple(1)),
        BatchTransform(lambda x: x.permute(2, 0, 1)),
        BatchTransform(PadToSquare(255)),
        BatchTransform(T.Resize((CROP_SIZE, CROP_SIZE))),
        BatchTransform(T.RandomResizedCrop(CROP_SIZE, scale=(0.8, 1), ratio=(1, 1))),
        BatchTransform(T.RandomHorizontalFlip()),
        BatchTransform(T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),),
        BatchTransform(T.RandomApply([T.GaussianBlur(kernel_size=224//20*2+1, sigma=(0.1, 2.0))], p=0.5),),
        ListToTensor(device, torch.float),
    ])
    sketch_transform = T.Compose([
        BatchTransform(SelectFromTuple(0)),
        BatchTransform(lambda x: x.permute(2, 0, 1)),
        BatchTransform(T.Resize((224, 224))),
        BatchTransform(RandomLineSkip(prob=0.5, skip=0.1)),
        BatchTransform(RandomRotation(prob=0.5, angle=30)),
        BatchTransform(T.RandomHorizontalFlip(p=0.5)),
        BatchTransform(T.RandomResizedCrop(224, scale=(0.8, 1), ratio=(1, 1))),
        ListToTensor(device, torch.float),
    ])

    # ENTRENAMIENTO
    
    learner = get_model(config)

    # se agregan las transformaciones a la red
    learner.augment1 = image_transform
    learner.augment2 = sketch_transform

    # optimizador
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    # base_optimizer = torch.optim.SGD(learner.parameters(), lr=0.1)
    # opt = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

    learner = learner.to(device)
    if LAST_EPOCH == 0:
        max_map = 0
    else:
        validation = EvalMAP(config, device)
        max_map = validation.compute_map(k=-1)
        print(f"\nmAP del último checkpoint: {max_map}")
    learner.train()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        running_loss = np.array([], dtype=np.float32)
        for epoch in range(LAST_EPOCH, TOTAL_EPOCHS):
            i = 0
            t0 = time.time()
            for images in train_loader:
                loss = learner(images)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if config['FRAMEWORK'] == 'BYOL':
                    learner.update_moving_average()
                running_loss = np.append(running_loss, [loss.item()])
                elapsed_time = time.time() - t0
                elapsed_timedelta = datetime.timedelta(seconds=elapsed_time)
                elapsed_timedelta = elapsed_timedelta - \
                    datetime.timedelta(microseconds=elapsed_timedelta.microseconds)
                elapsed_time_formatted = str(elapsed_timedelta)
                sys.stdout.write(
                    '\rEpoch {}, batch {} - loss {:.4f} - elapsed time {}'.format(epoch+1, i+1, np.mean(running_loss), elapsed_time_formatted))
                i += 1
            print_epoch_time(t0, epoch)
            # evaluamos el modelo con la data de test
            validation = EvalMAP(config, device, learner)
            current_map = validation.compute_map(k=-1)
            if current_map > max_map:
                max_map = current_map
                torch.save(learner.state_dict(), SAVE_PATH)
                print(f"\nnueva mAP máxima del modelo: {max_map}")
                configp['MODEL']['BEST_EPOCH'] = str(epoch + 1)
            else:
                print(f"\nmAP actual del modelo: {current_map} - mAP máxima del modelo: {max_map}")
            # torch.save(learner.state_dict(), SAVE_PATH)
            running_loss = np.array([], dtype=np.float32)
            configp['MODEL']['LAST_EPOCH'] = str(epoch + 1)
            with open(args.config, 'w') as configfile:
                configp.write(configfile)
            sys.stdout.write('\n')
    
