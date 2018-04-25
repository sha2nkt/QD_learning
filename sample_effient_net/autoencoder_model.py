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
            nn.Conv2d(12, 64,kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128,kernel_size=3,stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(True),
            # nn.Conv2d(128, 128, kernel_size=3, stride=2),
            # nn.ReLU(True),
        )
        self.encoder = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048)
        )
        self.forward_actions = nn.Sequential(
            nn.Linear(5, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128 * 7 * 7) #12 for 256
        )
        self.forward_deconv = nn.Sequential(
            # nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2),
            # nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2),
        )

    def forward(self, x, a):
        # TODO: Define forward pass
        x = self.forward_conv(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.encoder(x)
        a = self.forward_actions(a)
        x = x * a
        x = self.decoder(x)
        x = x.view(-1, 128, 7, 7)
        x = self.forward_deconv(x)
        return x