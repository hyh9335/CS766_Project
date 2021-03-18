#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from SRDataset import SRDataset
from torch.utils.data import DataLoader


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),#Pads the input tensor using the reflection of the input boundary.
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

def spectral_norm(module, mode=True): 
    #stabilizes training, typically used for discriminators, but also in EdgeGenerator in this paper. 
    if mode:
        return nn.utils.spectral_norm(module)
    return module
    
    


class Discriminator(nn.Module):
    #input (N,4,512,512), 4=[HR rgb(3) + HR or pred edge(1)]
    #return (N,1,H,W), where H, W are not necessarily 1.
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )
        
        for w in self.parameters():
            nn.init.normal_(w, 0, 0.02)


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

class EdgeGenerator(nn.Module):
    #input (N,4,512,512), 4=[LR rgb(3) + LR edge(1)], interpolated
    def __init__(self, scale=4, residual_blocks=8, use_spectral_norm=True):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )
        
        for w in self.parameters():
            nn.init.normal_(w, 0, 0.02)


    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

class AdversarialLoss(nn.Module):
    def __init__(self, type='hinge', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super().__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                #for discriminator
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                #for generator
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class EdgeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # generator input: [rgb(3) + edge(1)]
        # discriminator input: (rgb(3) + edge(1))
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator(in_channels=4, use_sigmoid=True)

        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)  #???

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR)/10,
            betas=(config.BETA1, config.BETA2)
        )
        
    def process(self, lr_images, hr_images, lr_edges, hr_edges):
        ## Update discriminator
        self.dis_optimizer.zero_grad()
        dis_loss = 0

        # process outputs from generator
        outputs = self(lr_images, lr_edges)

        # process outputs from generator
        dis_input_real = torch.cat((hr_images, hr_edges), dim=1)
        dis_input_fake = torch.cat((hr_images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (rgb(3) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (rgb(3) + edge(1))
        #discriminator loss
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        
        #logs
        logs = [
            ("l_dis", dis_loss.item()),
            ("dis_fake_loss", dis_fake_loss.item()),
            ("dis_real_loss", dis_real_loss.item())]
        
        #backward and update
        dis_loss.backward()
        self.dis_optimizer.step()


        
        ## Update generator       
        self.gen_optimizer.zero_grad()
        gen_loss = 0
        
        # process outputs from generator
        #Use the same output, since generator hasn't been updated yet
        
        # process outputs from updated discriminator
        gen_input_fake = torch.cat((hr_images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (rgb(3) + edge(1))
        """
        We cannot detach these two, because outputs lay behind them
        """

        #generator gan loss
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.ADV_LOSS_WEIGHT1
        gen_loss += gen_gan_loss
        # generator feature matching loss
        # using ground true, process outputs from updated discriminator
        dis_input_real = torch.cat((hr_images, hr_edges), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (rgb(3) + edge(1))    
        """
        Does this step need to be repeated again here???
        """
        
        gen_fm_loss = 0 
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss

        # create logs
        logs = logs+ [("l_gen", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]
        #backward and update
        gen_loss.backward()
        self.gen_optimizer.step()
        
        
        return outputs, 'gen_loss', dis_loss, logs        
    
    
    def forward(self, lr_images, lr_edges):
        hr_images = F.interpolate(lr_images, scale_factor=self.config.SCALE)
        hr_edges = F.interpolate(lr_edges, scale_factor=self.config.SCALE)
        inputs = torch.cat((hr_images, hr_edges), dim=1)
        outputs = self.generator(inputs)
        return outputs
        

        


import os
import yaml

class Config(dict):

    def __init__(self):
        '''
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)
            '''
 
    def __getattr__(self, name):
        '''
        if self._dict.get(name) is not None:
            return self._dict[name]

        if self.get(name) is not None:
            return self.get(name)
'''
        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


DEFAULT_CONFIG = {
    'MODE': 1,                      # 1: train, 2: test, 3: eval
    'MODEL': 1,                     # 1: edge model, 2: SR model, 3: SR model with edge enhancer
    'SCALE': 2,                  # scale factor (2, 4, 8)
    'SEED': 10,                     # random seed
    'GPU': [0],                     # list of gpu ids
    'DEBUG': 0,                     # turns on debugging mode
    'VERBOSE': 0,                   # turns on verbose mode in the output console

    'LR': 0.0001,                # learning rate
    'BETA1': 0.0,                # adam optimizer beta1
    'BETA2': 0.9,                # adam optimizer beta2
    'BATCH_SIZE': 2,                # input batch size for training
    'HR_SIZE': 256,                 # HR image size for training 0 for original size
    'SIGMA': 2,                     # standard deviation of the Gaussian filter used in Canny edge detector (0: random, -1: no edge)
    'MAX_ITERS': 2e7,               # maximum number of iterations to train the model
    'EDGE_THRESHOLD': 0.5,          # edge detection threshold

    'L1_LOSS_WEIGHT': 1,            # l1 loss weight
    'FM_LOSS_WEIGHT': 10,           # feature-matching loss weight
    'STYLE_LOSS_WEIGHT': 250,       # style loss weight
    'CONTENT_LOSS_WEIGHT': 0.1,     # content loss weight
    'ADV_LOSS_WEIGHT1': 0.1,        # edge model adversarial loss weight
    'ADV_LOSS_WEIGHT2': 1,          # SR model adversarial loss weight
    'GAN_LOSS': 'hinge',         # nsgan | lsgan | hinge

    'SAVE_INTERVAL': 1000,          # how many iterations to wait before saving model (0: never)
    'SAMPLE_INTERVAL': 1000,        # how many iterations to wait before sampling (0: never)
    'SAMPLE_SIZE': 12,              # number of images to sample
    'EVAL_INTERVAL': 0,             # how many iterations to wait before model evaluation (0: never)
    'LOG_INTERVAL': 10,             # how many iterations to wait before logging training status (0: never)
}




