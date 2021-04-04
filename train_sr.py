#!/usr/bin/env python
# coding: utf-8

from util.SRDataset import SRDataset
from util.metric import PSNR
from pytorch_msssim import ssim
from util.config import Config
from model import SRModel
import time
from torch.utils.data import DataLoader
import os

cfg = Config()
cfg.BATCH_SIZE = 8

model = SRModel(cfg).cuda()

sr_gen_path = "-".join(cfg.DATAPATH) + "gen_weights_path"
sr_disc_path = "-".join(cfg.DATAPATH) + "disc_weights_path"


import torch
try:
    if torch.cuda.is_available():
        data = torch.load(sr_gen_path)
        model.generator.load_state_dict(data['generator'])
        data = torch.load(sr_disc_path)
        model.discriminator.load_state_dict(data['discriminator'])
except Exception:
    # cannot read checkpoint
    pass

# maximum value of the picture is 1
psnr=PSNR(1.).cuda()
epochs = 10
set14 = SRDataset(os.path.join(*cfg.DATAPATH),
                  ["hr", "lr2x", "edge"])
train_loader = DataLoader(set14, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.BATCH_SIZE, pin_memory=True) #num_workers enables multi-process data loading

iterations = 0

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
            psnr_val = psnr(hr_images, hr_images_pred)
            logs.update({'psnr': psnr_val.item()})
            
            ssim_val = ssim(hr_images, hr_images_pred, data_range=1.)
            logs.update({"ssim": ssim_val.item()})

            time_end = time.time()
            logs.update({
                "epoch":  t,
                "iter": batch,
                "time cost": time_end - time_start
            }) 
            
            with open("logs.txt", "a", encoding='UTF-8') as f:
                f.write("\n"+"\t".join(i for i in sorted(logs)))
                f.write("\n"+"\t".join(str(round(logs[i],5)) for i in sorted(logs)))
            time_start = time.time()

        batch += 1

        iterations += 1

        if iterations % cfg.SAVE_INTERVAL == 0:
            torch.save({'generator': model.generator.state_dict()}, sr_gen_path)
            torch.save({'discriminator': model.discriminator.state_dict()}, sr_disc_path)
print("Done!")

