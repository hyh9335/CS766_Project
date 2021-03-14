import torch
from torch import nn
from util.init import dcgan_init


def spectral_norm_convs(do_spectral_norm, layer=None):
    """
        Converts conv layers to spectral norm.

        do_spectral_norm: whether to use spectral norm
        layer: the layer to convert, when it is None, it applies
            spectral norm a nn.Conv2d, with the bias removed, if you
            want to maintain the bias, use `spectral_norm(do_spectral_norm, nn.Conv2d)`

        Usage: 
            Conv2d = spectral_norm_convs(true, nn.Conv2d)
            m = Conv2d(...) # now m is a nn.Conv2d with spectral norm

    """
    def t(*args, **kwargs):
        if layer is None:
            m = nn.Conv2d(*args, **kwargs, bias=not do_spectral_norm)
        else:
            # not sure why it is not applied to ConvTranspose2d,
            # but leaving it as is for now
            m = layer(*args, **kwargs)
        
        if do_spectral_norm:
            return nn.utils.spectral_norm(m)
        else:
            return m

    return t


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, do_spectral_norm=True):
        super().__init__()
        Conv2d = spectral_norm_convs(do_spectral_norm)

        self.blocks = nn.Sequential(
            Conv2d(channels, channels, 3,
                 dilation=dilation,
                 padding=dilation, padding_mode='reflect'
                 ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            Conv2d(channels, channels, 3,
                 dilation=dilation,
                 padding=dilation, padding_mode='reflect'
                 ),
        )

    def forward(self, x):
        residual = x
        x = self.blocks(x)
        x += residual
        return x


class DCGANGenerator(nn.Module):
    def __init__(self, do_spectral_norm=True, net_type="edge"):
        """
            do_spectral_norm: whether to use spectral norm
                for conv layers, it is true for edge generating
                nets, and false for SR generator -- not sure why
            net_type: which the network for, "edge" for HR edge generating,
                and "sr" for generating HR image
        """
        super().__init__()
        Conv2d = spectral_norm_convs(do_spectral_norm)

        self.encoder = nn.Sequential(
            Conv2d(4, 64, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        )

        self.middle = nn.Sequential(
            *(
                ResidualBlock(channels=256, dilation=2, do_spectral_norm=do_spectral_norm)
                for _ in range(8)
            )
        )

        ConvTranspose2d = spectral_norm_convs(do_spectral_norm, nn.ConvTranspose2d)

        self.decoder = nn.Sequential(
            ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            Conv2d(64, 1 if net_type == "edge" else 3, 7, padding=3, padding_mode='reflect')
        )
        self.out = nn.Sigmoid()

        self.apply(dcgan_init)

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = self.out(x)
        return x
