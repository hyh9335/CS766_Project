import torch
from torch import nn
from util.init import dcgan_init


def spectral_norm_convs(use_spectral_norm, layer=None):
    """
        Converts conv layers to spectral norm.

        use_spectral_norm: whether to use spectral norm
        layer: the layer to convert, when it is None, it applies
            spectral norm a nn.Conv2d, with the bias removed, if you
            want to maintain the bias, use `spectral_norm(use_spectral_norm, nn.Conv2d)`

        Usage: 
            Conv2d = spectral_norm_convs(true, nn.Conv2d)
            m = Conv2d(...) # now m is a nn.Conv2d with spectral norm

    """
    def t(*args, **kwargs):
        if layer is None:
            m = nn.Conv2d(*args, **kwargs, bias=not use_spectral_norm)
        else:
            # not sure why it is not applied to ConvTranspose2d,
            # but leaving it as is for now
            m = layer(*args, **kwargs)
        
        if use_spectral_norm:
            return nn.utils.spectral_norm(m)
        else:
            return m

    return t


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, use_spectral_norm=True):
        super().__init__()
        Conv2d = spectral_norm_convs(use_spectral_norm)

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
    def __init__(self, use_spectral_norm=True, net_type="edge"):
        """
            use_spectral_norm: whether to use spectral norm
                for conv layers, it is true for edge generating
                nets, and false for SR generator -- not sure why
            net_type: which the network for, "edge" for HR edge generating,
                and "sr" for generating HR image
        """
        super().__init__()
        Conv2d = spectral_norm_convs(use_spectral_norm)

        self.net_type = net_type

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
                ResidualBlock(channels=256, dilation=2, use_spectral_norm=use_spectral_norm)
                for _ in range(8)
            )
        )

        ConvTranspose2d = spectral_norm_convs(use_spectral_norm, nn.ConvTranspose2d)

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
        if self.net_type == "sr":
            # not sure why the original code do tanh instead of sigmoid
            x = x * 2
        x = self.out(x)
        return x


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        Conv2d = spectral_norm_convs(use_spectral_norm)

        self.conv1 = nn.Sequential(
            Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.feature = self.conv1

        self.conv2 = nn.Sequential(
            Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            Conv2d(256, 512, 4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = nn.Sequential(
            Conv2d(512, 1, 4, stride=1, padding=1),
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
            outputs = torch.sigmoid(outputs)
        
        return outputs, [conv1, conv2, conv3, conv4, conv5]