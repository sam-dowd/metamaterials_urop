import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
from torch.optim.lr_scheduler import StepLR
from torch import autograd
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from typing import Dict, Tuple
from tqdm import tqdm
import numpy as np
import time
import os
import random
from tabulate import tabulate

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)
    
class Net4_Mode0Weight0_4x2real(nn.Module):
    """
    Net4 variant that predicts:
    - mode 0  (original index 0)
    - weight0 (original index 4)

    Output: [B, 2]
    Uses GroupNorm instead of BatchNorm.
    """
    def __init__(self):
        super().__init__()

        def groupnorm_2d(channels, num_groups=8):
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        def groupnorm_1d(features, num_groups=8):
            return nn.GroupNorm(num_groups=num_groups, num_channels=features)

        # ---------- CNN trunk ----------
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            groupnorm_2d(64), nn.GELU(),

            nn.Conv2d(64, 128, 3, 1, 1),
            groupnorm_2d(128), nn.GELU(),

            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, 1, 1),
            groupnorm_2d(256), nn.GELU(),

            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, 1, 1),
            groupnorm_2d(512), nn.GELU(),

            nn.MaxPool2d(2), nn.Dropout(0.25),

            Flatten()  # -> [B, 8192]
        )

        # ---------- Fully-connected head ----------
        self.fc = nn.Sequential(
            nn.Linear(8192 + 4, 2048),
            groupnorm_1d(2048), nn.GELU(), nn.Dropout(0.4),

            nn.Linear(2048, 1024),
            groupnorm_1d(1024), nn.GELU(), nn.Dropout(0.3),

            nn.Linear(1024, 256),
            groupnorm_1d(256), nn.GELU(), nn.Dropout(0.25),

            nn.Linear(256, 2)
        )

    def forward(self, x_img, x_cond):
        x = self.cnn(x_img)                  # [B, 8192]
        x = torch.cat((x, x_cond), dim=1)    # [B, 8196]
        return self.fc(x)                    # [B, 2]

class ResidualBlockGN(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.gn1   = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.gelu1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2   = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )

        self.activation = nn.GELU()

    def forward(self, x):
        identity = x

        out = self.gelu1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        return self.activation(out)

class Net4_Mode0Weight0_resnet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: [B, 1, 32, 32]
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )

        # ---------- Residual Blocks (10 conv layers total = 5 residual blocks) ----------
        self.res_blocks = nn.Sequential(
            ResidualBlockGN(64, 64),                # Conv 1–2
            ResidualBlockGN(64, 128, downsample=True),  # Conv 3–4, [B, 128, 16, 16]
            nn.Dropout(0.25),

            ResidualBlockGN(128, 256, downsample=True), # Conv 5–6, [B, 256, 8, 8]
            nn.Dropout(0.25),

            ResidualBlockGN(256, 512, downsample=True), # Conv 7–8, [B, 512, 4, 4]
            nn.Dropout(0.25),

            ResidualBlockGN(512, 512),              # Conv 9–10
        )

        self.flatten = Flatten()  # Output shape: [B, 512*4*4] = [B, 8192]

        # ---------- Fully Connected Head ----------
        def groupnorm_1d(features, num_groups=8):
            return nn.GroupNorm(num_groups=num_groups, num_channels=features)

        self.fc = nn.Sequential(
            nn.Linear(8192 + 4, 2048),
            groupnorm_1d(2048), nn.GELU(), nn.Dropout(0.4),

            nn.Linear(2048, 1024),
            groupnorm_1d(1024), nn.GELU(), nn.Dropout(0.3),

            nn.Linear(1024, 256),
            groupnorm_1d(256), nn.GELU(), nn.Dropout(0.25),

            nn.Linear(256, 2)
        )

    def forward(self, x_img, x_cond):
        x = self.stem(x_img)           # [B, 64, 32, 32]
        x = self.res_blocks(x)         # [B, 512, 4, 4]
        x = self.flatten(x)            # [B, 8192]
        x = torch.cat((x, x_cond), dim=1)  # [B, 8196]
        return self.fc(x)              # [B, 2]
