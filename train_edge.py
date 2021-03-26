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
train_loader = DataLoader(set14, batch_size=cfg.BATCH_SIZE, shuffle=True)



for t in range(epochs):
    print('\n\nTraining epoch: %d' % t)
    batch = 1
    for items in train_loader:
        time_start = time.time()
        #lr_images, hr_images, lr_edges, hr_edges=items
        lr_images, hr_images, lr_edges, hr_edges = (
            item.cuda() for item in items)
        hr_edges_pred, gen_loss, dis_loss, logs = model.process(
            lr_images, hr_images, lr_edges, hr_edges)

        
        time_end = time.time()
        logs = ["\n", ("epoch:", t), ("iter", batch),
                ('time cost', time_end - time_start)] + logs
        if batch % 10 == 0:
            precision, recall = edgeacc(hr_edges, hr_edges_pred)
            logs = ["\n", ("precision:", precision), ("recall", recall)] + logs
            with open("logs.txt", "a", encoding='UTF-8') as f:
                f.write("\n".join([str(i) for i in logs]))
        batch += 1
print("Done!")

'''To do: save trained model'''
