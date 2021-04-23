#!/usr/bin/env python
# coding: utf-8

import os
import sys

import torch
from util.SRDataset import SRDataset
from model import EdgeModel, SRModel
from util.config import Config

from util.metric import PSNR, EdgeAccuracy
from pytorch_msssim import ssim
from util.config import Config
from model import SRModel
import time
import torch
from torch.utils.data import DataLoader

def train(model_type, config):
    '''The process of training
        type: the network to be trained, 'edge' means only the edge model,
                'sr' means only the sr model, or 'both',
                raise a ValueError if it is incorrect
        model_type: the configuration, including the dataset, batch size, etc,
                should be of type util.Config

        A typical setup:

            config = Config()
            config.BATCH_SIZE = 8
            config.DATAPATH = ('dataset', 'celeba-hq')
            config.SCALE = 2
            train('both', config)
    '''
    if type(config) is not Config:
        raise TypeError('Expect `config` to be of type `util.Config`, got ', type(config), ' instead')

    if model_type == 'edge':
        train_edge(config)
    elif model_type == 'sr':
        train_sr(config)
    elif model_type == 'generate_edge':
        generate_edges(config)
    elif model_type == 'both':
        train_edge(config)
        generate_edges(config)
        train_sr(config)
    else:
        raise ValueError('Expect model_type to be one of `edge`, `sr` and `both`, ', 'got ', model_type)
    

def generate_edges(config):
    scale = config.SCALE
    edge_gen_path = os.path.join(*config.MODEL_PATH, "-".join(config.DATAPATH) + "_{0}x_".format(scale) 
        + "edge_gen_weights_path.pth")
    edge_disc_path = os.path.join(*config.MODEL_PATH, "-".join(config.DATAPATH) + "_{0}x_".format(scale) 
        + "edge_disc_weights_path.pth")

    model = EdgeModel(config).cuda()



    data = torch.load(edge_gen_path)
    model.generator.load_state_dict(data['generator'])
    data = torch.load(edge_disc_path)
    model.discriminator.load_state_dict(data['discriminator'])
    
    data = SRDataset(os.path.join(*config.DATAPATH),
                      ["hr", "lr{0}x".format(scale), "edge"])
    
    data.generate_image('pred_edge_lr{0}x'.format(scale), idx='all', model=model)


def train_edge(config):
    model = EdgeModel(config)
    edgeacc=EdgeAccuracy()

    scale = config.SCALE

    edge_gen_path = os.path.join(*config.MODEL_PATH, "-".join(config.DATAPATH) + "_{0}x_".format(scale) 
        + "edge_gen_weights_path.pth")
    edge_disc_path = os.path.join(*config.MODEL_PATH, "-".join(config.DATAPATH) + "_{0}x_".format(scale) 
        + "edge_disc_weights_path.pth")


    try:
        data = torch.load(edge_gen_path)
        model.generator.load_state_dict(data['generator'])
        data = torch.load(edge_disc_path)
        model.discriminator.load_state_dict(data['discriminator'])
        print("Loading checkpoint")
    except Exception:
        # cannot read checkpoint
        print("cannot read checkpoint")
        pass

    model.cuda()
    edgeacc.cuda()
        
    iterations = 1
    epochs = 10
    data = SRDataset(os.path.join(*config.DATAPATH),
                  ["lr{0}x".format(scale), "hr", "edge_lr{0}x".format(scale), "edge"],
                  img_list="train.csv")
    train_loader = DataLoader(data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    for t in range(epochs):
        print('\n\nTraining epoch: %d' % t)
        batch = 1
        time_start = time.time()
        for items in train_loader:

            #lr_images, hr_images, lr_edges, hr_edges=items
            lr_images, hr_images, lr_edges, hr_edges = (
                item.cuda(non_blocking=True) for item in items)
            hr_edges_pred, gen_loss, dis_loss, logs = model.process(
                lr_images, hr_images, lr_edges, hr_edges)



            if batch % 10 == 0:
                precision, recall = edgeacc(hr_edges, hr_edges_pred)
                logs.update({"precision:": precision.item(), "recall": recall.item()})
                time_end = time.time()
                logs.update ({"epoch:": t, "iter": batch,
                        'time cost': time_end - time_start})

                with open("-".join(config.DATAPATH) + "_{0}x_".format(scale) 
                    + "edge_logs.txt", "a", encoding='UTF-8') as f:
                    f.write("\n"+"\t".join(i for i in sorted(logs)))
                    f.write("\n"+"\t".join(str(round(logs[i],5)) for i in sorted(logs)))
                time_start = time.time()
        
            if iterations % config.SAVE_INTERVAL == 0:
                torch.save({'generator': model.generator.state_dict()}, edge_gen_path)
                torch.save({'discriminator': model.discriminator.state_dict()}, edge_disc_path)
            
            iterations += 1
            batch += 1
    
    print("Done!")


def train_sr(config):
    model = SRModel(config)

    scale = config.SCALE

    sr_gen_path = os.path.join(*config.MODEL_PATH, "-".join(config.DATAPATH) + "_{0}x_".format(scale) 
        + "sr_gen_weights_path.pth")
    sr_disc_path = os.path.join(*config.MODEL_PATH, "-".join(config.DATAPATH) + "_{0}x_".format(scale)
        + "sr_disc_weights_path.pth")


    try:
        data = torch.load(sr_gen_path)
        model.generator.load_state_dict(data['generator'])
        data = torch.load(sr_disc_path)
        model.discriminator.load_state_dict(data['discriminator'])
    except Exception:
        # cannot read checkpoint
        pass

    # maximum value of the picture is 1
    psnr=PSNR(1.)
    epochs = 10
    data = SRDataset(os.path.join(*config.DATAPATH),
                    ["hr", "lr{0}x".format(scale),
                     "pred_edge_lr{0}x".format(scale)],
                  img_list="train.csv")
    # num_workers=2 because colab only has 2
    train_loader = DataLoader(data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.BATCH_SIZE, pin_memory=True)
    iterations = 0

    model.cuda()

    psnr.cuda()
    for t in range(epochs):
        print('\n\nTraining epoch: %d' % t)
        batch = 1
        time_start = time.time()
        for items in train_loader:

            hr_images, lr_images, hr_edges = (
                item.cuda(non_blocking=True) for item in items)
            hr_images_pred, gen_loss, dis_loss, logs = model.process(
                lr_images, hr_images, hr_edges)

            if batch % 10 == 0:
                with torch.no_grad():
                    psnr_val = psnr(hr_images, hr_images_pred)
                logs.update({'psnr': psnr_val.item()})

                # ssim_val = ssim(hr_images, hr_images_pred, data_range=1.)
                # logs.update({"ssim": ssim_val.item()})

                time_end = time.time()
                logs.update({
                    "epoch":  t,
                    "iter": batch,
                    "time cost": time_end - time_start
                }) 

                with open("-".join(config.DATAPATH) + "_{0}x_".format(scale)
                     + "sr_logs.txt", "a", encoding='UTF-8') as f:
                    f.write("\n"+"\t".join(i for i in sorted(logs)))
                    f.write("\n"+"\t".join(str(round(logs[i],5)) for i in sorted(logs)))
                time_start = time.time()

            batch += 1

            iterations += 1

            if iterations % config.SAVE_INTERVAL == 0:
                torch.save({'generator': model.generator.state_dict()}, sr_gen_path)
                torch.save({'discriminator': model.discriminator.state_dict()}, sr_disc_path)
    print("Done!")
