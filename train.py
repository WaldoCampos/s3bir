import argparse
import configparser
import time
import datetime
import warnings
import sys
import os
import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
from models import get_model
from util.pairs_dataset import PairsDataset, pair_collate_fn
from transforms.custom_transforms import BatchTransform, SelectFromTuple, PadToSquare, ListToTensor


def print_epoch_time(t0):
    final_time = time.time() - t0
    final_timedelta = datetime.timedelta(seconds=final_time)
    final_timedelta = final_timedelta - datetime.timedelta(microseconds=final_timedelta.microseconds)
    final_time_formatted = str(final_timedelta)
    print(f"\nTiempo en epoch {epoch}: {final_time_formatted}")

if __name__ == '__main__':
    # Leemos el archivo de config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to the config file', required=True)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    config = config['MODEL']

    SAVE_PATH = config['SAVE_PATH']
    if not os.path.exists(os.path.split(SAVE_PATH)[0]):
        os.makedirs(os.path.split(SAVE_PATH)[0])

    device = "cuda:0"
    BATCH_SIZE = config.getint('BATCH_SIZE')
    CROP_SIZE = config.getint('CROP_SIZE')
    EPOCHS = config.getint('EPOCHS')
    TRAIN_DATA_DIR = config['TRAIN_DATA_DIR']
    torch.cuda.empty_cache()

    quickdraw = {"images": TRAIN_DATA_DIR,
            "sketches": TRAIN_DATA_DIR}
    quickdraw_dataset = PairsDataset(
        quickdraw["images"],
        quickdraw["sketches"]
    )
    train_loader = torch.utils.data.DataLoader(
        quickdraw_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pair_collate_fn,
        num_workers=config.getint('DATALOADER_WORKERS')
    )


    # resize the images for the net
    transforms_1 = T.Compose([
        BatchTransform(SelectFromTuple(0)),
        BatchTransform(PadToSquare(255)),
        BatchTransform(T.Resize((CROP_SIZE, CROP_SIZE))),
        BatchTransform(T.RandomResizedCrop(CROP_SIZE, scale=(0.8, 1), ratio=(1, 1))),
        ListToTensor(device, torch.float),
    ])
    transforms_2 = T.Compose([
        BatchTransform(SelectFromTuple(1)),
        BatchTransform(PadToSquare(255)),
        BatchTransform(T.Resize((CROP_SIZE, CROP_SIZE))),
        BatchTransform(T.RandomResizedCrop(CROP_SIZE, scale=(0.8, 1), ratio=(1, 1))),
        ListToTensor(device, torch.float),
    ])

    learner = get_model(config)

    # se agregan las transformaciones a la red
    learner.augment1 = transforms_1
    learner.augment2 = transforms_2

    # optimizador
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    learner = learner.to(device)
    learner.train()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        running_loss = np.array([], dtype=np.float32)
        for epoch in range(EPOCHS):
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
            print_epoch_time(t0)
            torch.save(learner.state_dict(), SAVE_PATH.format(epoch + 1))
            running_loss = np.array([], dtype=np.float32)
            sys.stdout.write('\n')
    