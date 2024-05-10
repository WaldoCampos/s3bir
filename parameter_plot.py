import os
import argparse
import configparser
import torch
import matplotlib.pyplot as plt
import numpy as np
from models import get_model

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    config = configparser.ConfigParser()
    path = '/home/wcampos/tests/s3bir/config/ecom_resnet50_{}.ini'
    param_numbers = []
    model_names = ['byol', 'simsiam', 'simclr', 'dino']
    for f in model_names:
        config.read(path.format(f))
        torch.cuda.empty_cache()
        learner = get_model(config['MODEL'])
        param_number = sum(p.numel() for p in learner.parameters())# if p.requires_grad)
        print(f"{f}: {param_number}")
        param_numbers.append(param_number)
    model_names = ['S3BIR-' + s.upper() for s in model_names]
    # MAP ecommerce
    maps_ecom = [0.2606, 0.2134, 0.4180, 0.2880, 0.4538]
    # MAP Flickr
    maps_flickr = [0.1176, 0.1086, 0.3424, 0.0907, 0.5403]

    # Agregamos datos de s3bir-clip
    param_numbers.append(149625345)
    model_names.append('S3BIR-CLIP')

    x = [n / 1000000 for n in param_numbers]
    y1 = maps_ecom
    y2 = maps_flickr
    types = model_names

    fig, ax = plt.subplots()#figsize=(7,5))
    # ax.scatter(x, y)

    ax.set_xlabel('Number of Parameters $(\\times10^6)$', fontsize=14)
    ax.set_ylabel('mAP', fontsize=14)
    ax.set(xlim=(30000001 / 1000000, 179999999 / 1000000), ylim=(0, 1))
    # ax.set_title('Rush Success Rate and EPA', fontsize=18)

    for i, txt in enumerate(types):
        ax.annotate(txt, (x[i], y1[i]), xytext=(5,-3), textcoords='offset points', fontsize=8)
        plt.scatter(x, y1, marker='s', color='red', label='eCommerce')
    for i, txt in enumerate(types):
        ax.annotate(txt, (x[i], y2[i]), xytext=(5,-3), textcoords='offset points', fontsize=8)
        plt.scatter(x, y2, marker='s', color='blue', label='Flickr15K')
    import matplotlib.patches as mpatches
    ecom = mpatches.Patch(color='red', label='eCommerce')
    flickr = mpatches.Patch(color='blue', label='Flickr15K')
    plt.legend(handles=[ecom, flickr])
    plt.savefig('/home/wcampos/param_plot.png')
