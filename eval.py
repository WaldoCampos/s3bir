import sys
import argparse
import configparser
import torch
import torchvision.transforms as T
import numpy as np
from models import get_model, get_dataset
from transforms.custom_transforms import PadToSquare, BatchTransform, SelectFromTuple, ListToTensor, RandomLineSkip, RandomRotation
import tensorflow_datasets as tfds
import hashlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_embeddings_labels(model, dataloader, device, mode):
    model.eval()
    embeddings = []
    labels = []
    for i, (batch, label) in enumerate(dataloader):
        batch = batch.to(device, dtype=torch.float)
        with torch.no_grad():
            current_embedding = model(batch, return_embedding=mode)
        embeddings.append(current_embedding.to('cpu'))
        labels.append(label)
        # sys.stdout.write('\rBatch {} done.'.format(i))
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0).numpy()


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


class EvalMAP():
    def __init__(self, config, device, learner):
        self.BATCH_SIZE = config.getint('BATCH_SIZE')
        self.CROP_SIZE = config.getint('CROP_SIZE')
        self.EPOCHS = config.getint('EPOCHS')
        self.DATASET = config['DATASET']
        self.DATALOADER_WORKERS = config.getint('DATALOADER_WORKERS')
        self.device = device

        torch.cuda.empty_cache()

        self.queries, self.catalogue = get_dataset(config, train=False)
        
        self.queries_loader = torch.utils.data.DataLoader(
            self.queries,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.DATALOADER_WORKERS,
            drop_last=True,
            )
        self.catalogue_loader = torch.utils.data.DataLoader(
            self.catalogue,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.DATALOADER_WORKERS,
            drop_last=True,
            )

        self.learner = learner.to(self.device)

        self.queries_embeddings, self.queries_labels = get_embeddings_labels(self.learner, self.queries_loader, self.device, 'target')
        self.catalogue_embeddings, self.catalogue_labels = get_embeddings_labels(self.learner, self.catalogue_loader, self.device, 'online')

        self.queries_embeddings = self.queries_embeddings / np.linalg.norm(self.queries_embeddings, ord=2, axis=1, keepdims=True)
        self.catalogue_embeddings = self.catalogue_embeddings / np.linalg.norm(self.catalogue_embeddings, ord=2, axis=1, keepdims=True)

        self.similarity_matrix = np.matmul(self.queries_embeddings, self.catalogue_embeddings.T)

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
    
    def create_file(self, k=-1):
        filename = '/home/wcampos/search_output.txt'
        file_out = open(filename, 'w')
        sorted_pos = np.argsort(-self.similarity_matrix, axis=1)
        sorted_pos_limited = sorted_pos[:, 1:] if k == -1 else sorted_pos[:, 1:k + 1]
        for i in np.arange(sorted_pos_limited.shape[0]):
            file_out.write(f"{self.queries_labels[i]}")
            predict = self.catalogue_labels[sorted_pos_limited[i,:]]
            for p in predict:
                file_out.write(f", {p}")
            file_out.write("\n")
        file_out.close()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Leemos el archivo de config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to the config file', required=True)
    parser.add_argument('--device', help='what device is to be used', required=True)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    config = config['MODEL']
    LAST_CHECKPOINT_PATH = config['LAST_CHECKPOINT_PATH']
    BEST_MAP_CHECKPOINT_PATH = config['BEST_MAP_CHECKPOINT_PATH']
    BEST_MAP5_CHECKPOINT_PATH = config['BEST_MAP5_CHECKPOINT_PATH']

    device = args.device

    torch.cuda.empty_cache()
    learner = get_model(config)

    learner.load_state_dict(torch.load(LAST_CHECKPOINT_PATH, map_location=torch.device(device)), strict=False)
    validation1 = EvalMAP(config, device, learner)
    learner.load_state_dict(torch.load(LAST_CHECKPOINT_PATH, map_location=torch.device(device)), strict=False)
    validation2 = EvalMAP(config, device, learner)
    learner.load_state_dict(torch.load(LAST_CHECKPOINT_PATH, map_location=torch.device(device)), strict=False)
    validation3 = EvalMAP(config, device, learner)
    k = -1
    final_metric = validation3.compute_map(k=k)

    print(f"\n{'mAP' if k==-1 else 'mAP@'+str(k)} del modelo: {final_metric}")