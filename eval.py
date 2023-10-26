import os
import sys
import argparse
import configparser
import torch
import torchvision.transforms as T
import numpy as np
from torchvision.datasets import ImageFolder
from models import get_model
from transforms.custom_transforms import PadToSquare
import tensorflow_datasets as tfds
import hashlib


def get_embeddings_labels(model, dataloader, mode):
    model.eval()
    embeddings = []
    labels = []
    for i, (batch, label) in enumerate(dataloader):
        batch = batch.to(device, dtype=torch.float)
        with torch.no_grad():
            current_embedding = model(batch, return_embedding=mode)
        embeddings.append(current_embedding.to('cpu'))
        labels.append(label)
        sys.stdout.write('\rBatch {} done.'.format(i))
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0).numpy()


def delete_duplicates_and_split(pairs_dataset):
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
            sketches.append((torch.from_numpy(s).permute(2, 0, 1), l))
        if p_hash not in unique_photos:
            unique_photos.add(p_hash)
            photos.append((torch.from_numpy(p).permute(2, 0, 1), l))
    return sketches, photos


class EvalMAP():
    def __init__(self, config, device, learner=None):
        self.BATCH_SIZE = config.getint('BATCH_SIZE')
        self.CROP_SIZE = config.getint('CROP_SIZE')
        self.EPOCHS = config.getint('EPOCHS')
        self.DATASET = config['DATASET']
        self.SAVE_PATH = config['SAVE_PATH']
        self.DATALOADER_WORKERS = config.getint('DATALOADER_WORKERS')

        torch.cuda.empty_cache()

        ds = tfds.load(self.DATASET,
            split='validation',
            as_supervised=True)
        # dataset = dataset.map(lambda x, y, _: (torch.from_numpy(x), torch.from_numpy(y)))
        ds = list(ds.as_numpy_iterator())

        queries, catalogue = delete_duplicates_and_split(ds)
        
        queries_loader = torch.utils.data.DataLoader(
            queries,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=self.DATALOADER_WORKERS,
            )
        catalogue_loader = torch.utils.data.DataLoader(
            catalogue,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=self.DATALOADER_WORKERS,
            )

        if learner == None:
            learner = get_model(config)

            learner.load_state_dict(torch.load(SAVE_PATH, map_location=torch.device(device)), strict=False)

        learner = learner.to(device)

        queries_embeddings, self.queries_labels = get_embeddings_labels(learner, queries_loader, 'target')
        catalogue_embeddings, self.catalogue_labels = get_embeddings_labels(learner, catalogue_loader, 'online')

        queries_embeddings = queries_embeddings / np.linalg.norm(queries_embeddings, ord=2, axis=1, keepdims=True)
        catalogue_embeddings = catalogue_embeddings / np.linalg.norm(catalogue_embeddings, ord=2, axis=1, keepdims=True)

        self.similarity_matrix = np.matmul(queries_embeddings, catalogue_embeddings.T)

    def compute_map(self, k=5):
        sorted_pos = np.argsort(-self.similarity_matrix, axis = 1)                
        AP = []
        sorted_pos_limited = sorted_pos[:, 1:] if k == -1 else sorted_pos[:, 1:k + 1]
        for i in np.arange(sorted_pos_limited.shape[0]):
            ranking = self.catalogue_labels[sorted_pos_limited[i,:]]                 
            pos_query = np.where(ranking == self.queries_labels[i])[0]
            pos_query = pos_query + 1 
            if len(pos_query) == 0 :
                AP_q = 0
            else :
                recall = np.arange(1, len(pos_query) + 1)
                pr = recall / pos_query
                AP_q = np.mean(pr)
            AP.append(AP_q)                            
        mAP = np.mean(np.array(AP))        
        return mAP


if __name__ == '__main__':
    # Leemos el archivo de config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to the config file', required=True)
    parser.add_argument('--device', help='what device is to be used', required=True)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    config = config['MODEL']

    device = args.device

    BATCH_SIZE = config.getint('BATCH_SIZE')
    CROP_SIZE = config.getint('CROP_SIZE')
    EPOCHS = config.getint('EPOCHS')
    DATASET = config['DATASET']
    SAVE_PATH = config['SAVE_PATH']

    torch.cuda.empty_cache()

    validation = EvalMAP(config, device=device)

    k = -1
    final_metric = validation.compute_map(k=k)

    print(f"\n{'mAP' if k==-1 else 'mAP@'+str(k)} del modelo: {final_metric}")