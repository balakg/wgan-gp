import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    """2D Conv with activation."""
    def __init__(self, in_dim, out_dim, ksize, stride, padding):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=padding),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, z_dim, im_size):
        super(Generator, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(0.2, inplace=True))

        n_levels = int(np.log2(im_size))
        cur_dim = z_dim
        layers = []
        for i in range(n_levels):
            layers.append(ConvBlock(cur_dim, max(64, cur_dim//2), 3, 1, 1))
            layers.append(nn.Upsample(scale_factor=2))
            cur_dim = max(64, cur_dim//2)
 
        self.decoder =  nn.Sequential(*layers)
        self.conv = nn.Conv2d(cur_dim, 1, kernel_size=7, stride=1, padding=3)

 
    def forward(self, z):
        x = self.shared(z)   #Fully-connected layers
        x = x.view(x.size(0), -1, 1, 1)  #Reshape to a 1x1 image
        x = self.decoder(x) #Decode back to image size
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, im_size):
        super(Discriminator, self).__init__()
        layers = []
        cur_dim, next_dim = 1, 64
        repeat_num = int(np.log2(im_size/4))

        for i in range(repeat_num):
            layers.append(ConvBlock(cur_dim, next_dim, 4, 2, 1))
            cur_dim, next_dim = next_dim, min(256, next_dim*2)

        layers.append(nn.Conv2d(cur_dim, 1, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*layers)

 
    def forward(self, x):
        return self.main(x)
