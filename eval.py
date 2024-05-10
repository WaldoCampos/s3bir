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
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_embeddings_labels(model, dataloader, device, mode, QDEXT=False):
    model.eval()
    embeddings = []
    labels = []
    image_transform = T.Compose([
                        BatchTransform(lambda x: torch.from_numpy(x)),
                        BatchTransform(lambda x: x.permute(2, 0, 1)),
                        BatchTransform(PadToSquare(255)),
                        BatchTransform(T.Resize((224, 224))),
                        BatchTransform(lambda x: x / 255),
                        ListToTensor(device, torch.float),
                    ])
    for i, (batch, label) in enumerate(dataloader):
        if QDEXT:
            batch = image_transform(batch)
            label = torch.from_numpy(label).to('cpu')
        batch = batch.to(device, dtype=torch.float)
        with torch.no_grad():
            current_embedding = model(batch, return_embedding=mode)
        embeddings.append(current_embedding.to('cpu'))
        labels.append(label)
        sys.stdout.write('\rBatch {} done.'.format(i))
    sys.stdout.write('\n')
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
        self.DATASET = config['DATASET']
        QDEXT = False
        if self.DATASET in ['QDEXT', 'QDEXT_UNKNOWN']:
            QDEXT = True
        self.DATALOADER_WORKERS = config.getint('DATALOADER_WORKERS')
        self.device = device

        torch.cuda.empty_cache()

        self.queries, self.catalogue = get_dataset(config, train=False)
        
        self.queries_loader = torch.utils.data.DataLoader(
            self.queries,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.DATALOADER_WORKERS,
            # drop_last=True,
            )
        self.catalogue_loader = torch.utils.data.DataLoader(
            self.catalogue,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.DATALOADER_WORKERS,
            # drop_last=True,
            )

        self.learner = learner.to(self.device)

        if QDEXT:
            self.queries_embeddings, self.queries_labels = get_embeddings_labels(self.learner, self.queries, self.device, 'target', QDEXT)
            self.catalogue_embeddings, self.catalogue_labels = get_embeddings_labels(self.learner, self.catalogue, self.device, 'online', QDEXT)
        else:
            self.queries_embeddings, self.queries_labels = get_embeddings_labels(self.learner, self.queries_loader, self.device, 'target', QDEXT)
            self.catalogue_embeddings, self.catalogue_labels = get_embeddings_labels(self.learner, self.catalogue_loader, self.device, 'online', QDEXT)

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

    def create_rankings(self):
        sorted_pos = np.argsort(-self.similarity_matrix, axis=1)
        ordered_relevants = []
        for i in np.arange(sorted_pos.shape[0]):
            truth = self.queries_labels[i]
            predict = self.catalogue_labels[sorted_pos[i,:]]
            ordered_relevants.append((predict == truth).astype(int))
        return np.stack(ordered_relevants)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

    device = args.device

    torch.cuda.empty_cache()
    learner = get_model(config)

    learner.load_state_dict(torch.load(BEST_MAP_CHECKPOINT_PATH, map_location=torch.device(device)), strict=False)
    validation = EvalMAP(config, device, learner)
    k = 200
    final_metric = validation.compute_map(k=k)

    print(f"\n{'mAP' if k==-1 else 'mAP@'+str(k)} del modelo: {final_metric}")
