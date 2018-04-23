import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pdb

class PredictorNet(nn.Module):
    def __init__(self, num_classes=20):
        super(PredictorNet, self).__init__()
        # TODO: Define model
        self.features = nn.Sequential(
            nn.Linear(81, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True,)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 81)
        )

    def forward(self, x1, x2):
        # TODO: Define forward pass
        x1 = self.features(x1)
        x2 = self.features(x2)
        x3 = torch.cat((x1,x2), dim=1)
        x3 = self.classifier(x3)
        return x3