import torch
from torch import nn


def spectral_norm(m: nn.Module):
    if type(m) is nn.Conv2d:
        # print(m)
        nn.utils.spectral_norm(m)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(channels, channels, 3,
                      dilation=dilation,
                      padding=dilation, padding_mode='reflect'
                      ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3,
                      dilation=dilation,
                      padding=dilation, padding_mode='reflect'
                      ),
        )

    def forward(self, x):
        residual = x
        x = self.blocks(x)
        x += residual
        return x


class Generator(nn.Module):
    def __init__(self, do_spectral_norm=True):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        )

        self.middle = nn.Sequential(
            *(
                ResidualBlock(channels=256, dilation=2)
                for _ in range(8)
            )
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 1, 7, padding=3, padding_mode='reflect')
        )
        self.out = nn.Sigmoid()

        if do_spectral_norm:
            self.apply(spectral_norm)

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = self.out(x)
        return x