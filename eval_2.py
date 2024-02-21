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


def compute_map(similarity_matrix, catalogue_labels, queries_labels, k=5):
        sorted_pos = np.argsort(-similarity_matrix, axis = 1)                
        AP = []
        sorted_pos_limited = sorted_pos[:, 1:] if k == -1 else sorted_pos[:, 1:k + 1]
        for i in np.arange(sorted_pos_limited.shape[0]):
            ranking = catalogue_labels[sorted_pos_limited[i,:]]                 
            pos_query = np.where(ranking == queries_labels[i])[0]
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
    

def create_file(model, device, catalogue_dataloader, queries_dataloader, k):
        queries_embeddings, queries_labels = get_embeddings_labels(model, queries_dataloader, device, 'target')
        catalogue_embeddings, catalogue_labels = get_embeddings_labels(model, catalogue_dataloader, device, 'online')

        queries_embeddings = queries_embeddings / np.linalg.norm(queries_embeddings, ord=2, axis=1, keepdims=True)
        catalogue_embeddings = catalogue_embeddings / np.linalg.norm(catalogue_embeddings, ord=2, axis=1, keepdims=True)

        similarity_matrix = np.matmul(queries_embeddings, catalogue_embeddings.T)
        final_metric = compute_map(similarity_matrix, catalogue_labels, queries_labels, k=k)
        print(f"mAP del modelo: {final_metric}")
        filename = '/home/wcampos/search_output.txt'
        file_out = open(filename, 'w')
        sorted_pos = np.argsort(-similarity_matrix, axis=1)
        sorted_pos_limited = sorted_pos[:, 1:] if k == -1 else sorted_pos[:, 1:k + 1]
        for i in np.arange(sorted_pos_limited.shape[0]):
            file_out.write(f"{queries_labels[i]}")
            predict = catalogue_labels[sorted_pos_limited[i,:]]
            for p in predict:
                file_out.write(f",{p}")
            file_out.write("\n")
        file_out.close()


def eval_model(model, device, catalogue_dataloader, queries_dataloader, k):
    queries_embeddings, queries_labels = get_embeddings_labels(model, queries_dataloader, device, 'target')
    catalogue_embeddings, catalogue_labels = get_embeddings_labels(model, catalogue_dataloader, device, 'online')

    queries_embeddings = queries_embeddings / np.linalg.norm(queries_embeddings, ord=2, axis=1, keepdims=True)
    catalogue_embeddings = catalogue_embeddings / np.linalg.norm(catalogue_embeddings, ord=2, axis=1, keepdims=True)

    similarity_matrix = np.matmul(queries_embeddings, catalogue_embeddings.T)
    final_metric = compute_map(similarity_matrix, catalogue_labels, queries_labels, k=k)
    return final_metric


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

    device = args.device

    torch.cuda.empty_cache()
    learner = get_model(config)

    learner.load_state_dict(torch.load(BEST_MAP_CHECKPOINT_PATH, map_location=torch.device(device)), strict=False)
    BATCH_SIZE = config.getint('BATCH_SIZE')
    CROP_SIZE = config.getint('CROP_SIZE')
    EPOCHS = config.getint('EPOCHS')
    DATASET = config['DATASET']
    DATALOADER_WORKERS = config.getint('DATALOADER_WORKERS')

    torch.cuda.empty_cache()

    queries, catalogue = get_dataset(config, train=False)
    
    queries_loader = torch.utils.data.DataLoader(
        queries,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        drop_last=True,
        )
    catalogue_loader = torch.utils.data.DataLoader(
        catalogue,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        drop_last=True,
        )

    learner = learner.to(device)
    k=-1
    final_metric = eval_model(learner, device, catalogue_loader, queries_loader, k)
    print(f"\n{'mAP' if k==-1 else 'mAP@'+str(k)} del modelo: {final_metric}")
    create_file(learner, device, catalogue_loader, queries_loader, k)