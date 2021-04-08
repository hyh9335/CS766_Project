# most comes from github.com/knazeri/edge-informed-sisr, with modification to optimizer steps

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from networks import DCGANGenerator, PatchGANDiscriminator
from loss import AdversarialLoss, StyleContentLoss
import numpy as np


class EdgeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # generator input: [rgb(3) + edge(1)]
        # discriminator input: (rgb(3) + edge(1))
        generator = DCGANGenerator(use_spectral_norm=True, net_type="edge")
        discriminator = PatchGANDiscriminator(in_channels=4, use_sigmoid= config.GAN_LOSS != 'hinge')

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
            lr=float(config.LR),                                     
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
        dis_real_loss = self.adversarial_loss(dis_real, True, True)         # loss=1~0 if dis_real=0~1; 
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)        # loss=1~2 if dis_real=0~1; 
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        
        #logs
        logs = {
            "l_dis": dis_loss.item(),
            "dis_fake_loss": dis_fake_loss.item(),
            "dis_real_loss": dis_real_loss.item()}
        
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
        # generator feature matching loss                                   #loss=0~-1 if gen_fake=0~1; 
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
        logs.update({"l_gen": gen_gan_loss.item(),
            "l_fm": gen_fm_loss.item(),
            "l_gen_total": gen_loss.item()})

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


class SRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # generator input: [rgb(3) + edge(1)]
        # discriminator input: rgb(3)
        self.generator = DCGANGenerator(use_spectral_norm=False, net_type="sr")
        self.discriminator = PatchGANDiscriminator(in_channels=3, use_sigmoid= config.GAN_LOSS != 'hinge')

        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.style_content_loss = StyleContentLoss()

        kernel = np.zeros((self.config.SCALE, self.config.SCALE))
        kernel[0, 0] = 1

         # (out_channels, in_channels/groups, height, width)
        scale_kernel = torch.FloatTensor(np.tile(kernel, (3, 1, 1, 1,)))
        self.register_buffer('scale_kernel', scale_kernel)

        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=float(config.LR),                                        
            betas=(config.BETA1, config.BETA2)                          
        )

        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(config.LR),                                     
            betas=(config.BETA1, config.BETA2)
        )
    
    def dis_step(self, outputs, lr_images, hr_images, hr_edges):
        ## Update discriminator
        self.dis_optimizer.zero_grad()
        dis_loss = 0

        # process outputs from generator
        dis_input_fake = outputs.detach_()
        dis_real, dis_real_feat = self.discriminator(hr_images)
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)
        #discriminator loss
        dis_real_loss = self.adversarial_loss(dis_real, True, True)         # loss=1~0 if dis_real=0~1; 
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)        # loss=1~2 if dis_real=0~1; 
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        
        #backward and update
        dis_loss.backward()
        self.dis_optimizer.step()

        return dis_loss.item(), dict([("l_dis", dis_loss.item())])
    
    def gen_step(self, outputs, lr_images, hr_images, hr_edges):
        ## Update generator       
        self.gen_optimizer.zero_grad()
        gen_loss = 0
        
        # process outputs from generator
        #Use the same output, since generator hasn't been updated yet
        
        # process outputs from updated discriminator
        gen_input_fake = outputs
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)

        #generator gan loss
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.ADV_LOSS_WEIGHT2
        gen_loss += gen_gan_loss

        
        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, hr_images) * self.config.L1_LOSS_WEIGHT
        gen_loss += gen_l1_loss


        # generator content & style loss
        gen_style_loss, gen_content_loss = self.style_content_loss(outputs, hr_images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_content_loss
        gen_loss += gen_style_loss

        # using ground true, process outputs from updated discriminator
        dis_input_real = hr_images
        dis_real, dis_real_feat = self.discriminator(dis_input_real)  
        
        gen_fm_loss = 0 
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss

        # create logs
        logs = [
            ("l_gen", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_content", gen_content_loss.item()),
            ("l_style", gen_style_loss.item()),
        ]

        logs = dict(logs)

        self.gen_optimizer.step()

        return gen_loss.item(), logs

    def process(self, lr_images, hr_images, hr_edges):
        
        # process outputs from generator
        outputs = self(lr_images, hr_edges)

        logs = {}

        dis_loss, dlogs = self.dis_step(outputs, lr_images, hr_images, hr_edges)
        gen_loss, glogs = self.gen_step(outputs, lr_images, hr_images, hr_edges)

        logs.update(dlogs)
        logs.update(glogs)

        return outputs, dis_loss, gen_loss, logs


    def forward(self, lr_images, hr_edges):
        hr_images = F.conv_transpose2d(lr_images, self.scale_kernel, stride=2, groups=3)
        inputs = torch.cat((hr_images, hr_edges), dim=1)
        outputs = self.generator(inputs)
        return outputs