import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, in_channels=100, channels=64, out_channels=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels, channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(channels * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(channels * 8, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(channels, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ngpu, in_channels=3, channels=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels, channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(channels * 2, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(channels * 4, channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(channels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    x = torch.randn(4, 100, 1, 1)
    model = Generator(ngpu=1)
    y = model(x)
    print(y.size())
