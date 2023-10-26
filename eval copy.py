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


def compute_map(sim_matrix, q_labels, c_labels, k=5):
    sorted_pos = np.argsort(-sim_matrix, axis = 1)                
    AP = []
    sorted_pos_limited = sorted_pos[:, 1:] if k == -1 else sorted_pos[:, 1:k + 1]
    for i in np.arange(sorted_pos_limited.shape[0]):
        ranking = c_labels[sorted_pos_limited[i,:]]                 
        pos_query = np.where(ranking == q_labels[i])[0]
        pos_query = pos_query + 1 
        if len(pos_query) == 0 :
            AP_q = 0
        else :
            recall = np.arange(1, len(pos_query) + 1)
            pr = recall / pos_query
            AP_q = np.mean(pr)
        AP.append(AP_q)
        #print('{} -> mAP = {}'.format(len(pos_query), AP_q))
                        
    mAP = np.mean(np.array(AP))        
    return mAP


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
    TEST_CATALOGUE_DIR = config['TEST_CATALOGUE_DIR']
    TEST_QUERY_DIR = config['TEST_QUERY_DIR']
    SAVE_PATH = config['SAVE_PATH']

    torch.cuda.empty_cache()

    #Transformaciones para las imágenes
    transform_queries = T.Compose([
        PadToSquare(fill=255),
        T.Resize((224,224)),
        T.PILToTensor()
    ])

    transform_catalogue = T.Compose([
        PadToSquare(fill=255),
        T.Resize((224,224)),
        T.PILToTensor()
    ])

    #Cargar archivos
    quickdraw = {
        "images": TEST_CATALOGUE_DIR,
        "sketches": TEST_QUERY_DIR,
        "red": SAVE_PATH,
    }
    queries = ImageFolder(
        root = quickdraw["sketches"],
        transform = transform_queries)

    catalogue = ImageFolder(
        root = quickdraw["images"],
        transform = transform_catalogue)


    queries_loader = torch.utils.data.DataLoader(queries, batch_size=BATCH_SIZE, shuffle=False)
    catalogue_loader = torch.utils.data.DataLoader(catalogue, batch_size=BATCH_SIZE, shuffle=False)

    learner = get_model(config)

    learner.load_state_dict(torch.load(SAVE_PATH, map_location=torch.device(device)), strict=False)

    learner = learner.to(device)

    queries_embeddings, queries_labels = get_embeddings_labels(learner, queries_loader, 'target')
    catalogue_embeddings, catalogue_labels = get_embeddings_labels(learner, catalogue_loader, 'online')

    queries_embeddings = queries_embeddings / np.linalg.norm(queries_embeddings, ord=2, axis=1, keepdims=True)
    catalogue_embeddings = catalogue_embeddings / np.linalg.norm(catalogue_embeddings, ord=2, axis=1, keepdims=True)

    similarity_matrix = np.matmul(queries_embeddings, catalogue_embeddings.T)

    final_metric = compute_map(similarity_matrix, queries_labels, catalogue_labels, k=-1)

    print(f"mAP@5 del modelo: {final_metric}")