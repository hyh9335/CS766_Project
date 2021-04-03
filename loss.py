import torch
from torch import nn
import torchvision.models as models



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
                # for discriminator
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                # for generator
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleContentLoss(nn.Module):
    r"""
    Returns a tuple of (style, content) loss
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    style_loss_layers = ('relu2_2', 'relu3_4', 'relu4_4', 'relu5_2')
    content_loss_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

    def __init__(self):
        super().__init__()
        self.vgg_features = VGG19Features()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg_features(x), self.vgg_features(y)

        # Compute loss
        style_loss = 0.0
        for layer in self.__class__.style_loss_layers:
            style_loss += self.criterion(self.compute_gram(x_vgg[layer]), self.compute_gram(y_vgg[layer]))

        content_loss = 0.0
        for num, layer in enumerate(self.__class__.content_loss_layers):
            content_loss += self.weights[num] * self.criterion(x_vgg[layer], y_vgg[layer])
        
        return style_loss, content_loss


class VGG19Features(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = vgg19.features[:36]
        self.names_map = {
            1: 'relu1_1',
            3: 'relu1_2',

            6: 'relu2_1',
            8: 'relu2_2',

            11: 'relu3_1',
            13: 'relu3_2',
            15: 'relu3_3',
            17: 'relu3_4',

            20: 'relu4_1',
            22: 'relu4_2',
            24: 'relu4_3',
            26: 'relu4_4',

            29: 'relu5_1',
            31: 'relu5_2',
            33: 'relu5_3',
            35: 'relu5_4',
        }

        # use forward hook to obtain features
        def get_feature(self, name):
            # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
            def hook(model, inp, out):
                self.outputs[name] = out.detach()
            return hook

        for k in self.names_map:
            self.features[k].register_forward_hook(get_feature(self, self.names_map[k]))

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input):
        self.outputs = {}
        with torch.no_grad():
            self.features(input)
        return self.outputs
