import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pdb

class PredictorNet(nn.Module):
    def __init__(self, num_classes=20):
        super(PredictorNet, self).__init__()
        # TODO: Define model
        self.forward_conv = nn.Sequential(
            # Input: 256x256
            nn.Conv2d(12, 128,kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128,kernel_size=4,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.encoder = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

        )
        self.forward_actions = nn.Sequential(
            nn.Linear(6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )

        self.reward = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256,3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 128 * 6 * 6), #12 for 256
            nn.BatchNorm1d(128 * 6 * 6),
            nn.ReLU(True),
        )
        self.forward_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x, a):
        # TODO: Define forward pass
        x = self.forward_conv(x)
        x = x.view(-1, 128 * 6 * 6)
        x = self.encoder(x)
        a = self.forward_actions(a)
        x = x * a
        r = self.reward(x)
        x = self.decoder(x)
        x = x.view(-1, 128, 6, 6)
        x = self.forward_deconv(x)
        return x, r




class Discriminator(nn.Module):
    def __init__(self, num_classes=20):
        super(Discriminator, self).__init__()
        # TODO: Define model
        self.forward_conv = nn.Sequential(
            # Input: 256x256
            nn.Conv2d(3, 128,kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128,kernel_size=4,stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # TODO: Define forward pass
        x = self.forward_conv(x)
        x = x.view(-1, 128 * 11 * 8)
        x = self.discriminator(x)
        return x
