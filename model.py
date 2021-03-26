import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from networks import DCGANGenerator, PatchGANDiscriminator

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
        generator = DCGANGenerator(use_spectral_norm=True, net_type="edge")
        discriminator = PatchGANDiscriminator(in_channels=4, use_sigmoid=True)

        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)  #???

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR*10),                                        #The learning rate if discriminator is speed up!
            betas=(config.BETA1, config.BETA2)                          
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR)/10,                                     #The learning rate if discriminator is slowed down!!
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