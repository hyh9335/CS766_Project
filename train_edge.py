#!/usr/bin/env python
# coding: utf-8


from util.SRDataset import SRDataset
from util.metric import EdgeAccuracy
from util.config import Config
from model import EdgeModel
import time
from torch.utils.data import DataLoader
import os

cfg = Config()
model = EdgeModel(cfg).cuda()
# model = EdgeModel(cfg)
edgeacc=EdgeAccuracy().cuda()
epochs = 10
set14 = SRDataset(os.path.join(*cfg.DATAPATH),
                  ["lr2x", "hr", "edge_lr2x", "edge"])
train_loader = DataLoader(set14, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2,pin_memory=True) #num_workers enables multi-process data loading





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
            
            with open("logs.txt", "a", encoding='UTF-8') as f:
                f.write("\n"+"\t".join(i for i in sorted(logs)))
                f.write("\n"+"\t".join(str(round(logs[i],5)) for i in sorted(logs)))
            time_start = time.time()
        if batch % 100 == 0:
          torch.save({'generator': model.generator.state_dict()}, "gen_weights_path")
          torch.save({'discriminator': model.discriminator.state_dict()},"dis_weights_path")
        batch += 1
print("Done!")



import torch
if torch.cuda.is_available():
    data = torch.load("2nd_gen_weights_path")
model.generator.load_state_dict(data['generator'])

if torch.cuda.is_available():
    data = torch.load("2nd_dis_weights_path")

model.discriminator.load_state_dict(data['discriminator'])