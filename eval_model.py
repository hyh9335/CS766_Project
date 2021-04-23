import os
import torch
from util.SRDataset import SRDataset
import time
from util.metric import PSNR
from model import FullModel
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

def generate_pred(config):
    full_model = FullModel(config)
 
    full_model.eval()
    full_model.cuda()

    data = SRDataset(os.path.join(*config.DATAPATH),
            ["lr{0}x".format(config.SCALE), "edge_lr{0}x".format(config.SCALE), "hr"],
            img_list="eval.csv")

    data.generate_image('pred_full_lr{}x'.format(config.SCALE), model=full_model, no_single_point=False)

def eval_full(config, generate=True):
    if generate:
        generate_pred(config)
    
    # maximum value of the picture is 1
    psnr=PSNR(1.)

    data = SRDataset(os.path.join(*config.DATAPATH),
            ["pred_full_lr{0}x".format(config.SCALE), "hr"],
            img_list="eval.csv")

    eval_loader = DataLoader(data, batch_size=config.BATCH_SIZE * 2, shuffle=True, num_workers=config.BATCH_SIZE * 2, pin_memory=True)

    psnr.cuda()

    num_batches = 0

    tot_psnr = 0
    tot_ssim = 0
    
    for items in eval_loader:
        num_batches += 1

        pred_hr_images, hr_images= (
            item.cuda(non_blocking=True) for item in items)

        with torch.no_grad():
            tot_psnr = psnr(pred_hr_images, hr_images) + tot_psnr
            tot_ssim = ssim(pred_hr_images, hr_images) + tot_ssim
        
    
    print("PSNR: ", tot_psnr / num_batches)
    print("SSIM: ", tot_ssim / num_batches)
