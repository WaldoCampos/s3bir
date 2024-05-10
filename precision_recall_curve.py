import os
import argparse
import configparser
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from eval import EvalMAP
from models import get_model

def compute_p11(ranking):
    p_11 = np.zeros(11)
    n_queries = ranking.shape[0]
    n_catalog = ranking.shape[1]
    n_relevants = np.sum(ranking, axis=1)
    for q in np.arange(n_queries):
        if n_relevants[q] > 0:
            positions = np.arange(n_catalog) + 1
            relevants_idx_q = ranking[q]
            relevants_idx_q_cum = np.cumsum(relevants_idx_q)
            recall = relevants_idx_q_cum / n_relevants[q]
            precision = relevants_idx_q_cum / positions
            p = precision[relevants_idx_q==1]
            for r in np.arange(11):
                p_11[r] += np.max(precision[recall >= r*0.1])
    return p_11/n_queries

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    prefix = '/home/wcampos/tests/s3bir/config/'
    config_list = [
        prefix + 'ecom_resnet50_byol.ini',
        prefix + 'ecom_resnet50_simsiam.ini',
        prefix + 'ecom_resnet50_simclr.ini',
        prefix + 'ecom_resnet50_dino.ini',
    ]

    r11 = np.arange(0,11,1)/10
    for c in config_list:
        label = 's3bir-' + c.split('/')[-1].split('.')[0].split('_')[-1]
        config = configparser.ConfigParser()
        config.read(c)
        config = config['MODEL']
        BEST_MAP_CHECKPOINT_PATH = config['BEST_MAP_CHECKPOINT_PATH']

        device = 'cuda:0'

        torch.cuda.empty_cache()
        learner = get_model(config)

        learner.load_state_dict(torch.load(BEST_MAP_CHECKPOINT_PATH, map_location=torch.device(device)), strict=False)
        validation = EvalMAP(config, device, learner)
        ranking = validation.create_rankings()
        p11 = compute_p11(ranking)
        plt.plot(r11, p11, 'o-', label=label)

    # cargamos los binarios de clip-sbir y los agregamos al plot
    ranking = np.load('/home/data/s3bir/binarios/sbirclip_ecommerce.pkl', allow_pickle=True)
    # ranking = np.load('/home/data/s3bir/binarios/sbirclip_flickr15k.pkl', allow_pickle=True)
    p11 = compute_p11(ranking)
    plt.plot(r11, p11, 'o-', label='s3bir-clip')

    plt.yticks(ticks=np.arange(0,11,1)/10)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.grid(True, axis = 'y')    
    plt.grid(True, axis = 'x')
    plt.legend()
    # plt.show()
    plt.savefig('/home/wcampos/ecom_precision_recall.png')