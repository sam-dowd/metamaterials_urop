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

class Net4_ModeOnly_TiledConditioning(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------- Initial Conv Layers (before tiling) ----------
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)   # → [B, 64, 32, 32]
        self.gn1   = nn.GroupNorm(8, 64)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1) # → [B, 128, 16, 16]
        self.gn2   = nn.GroupNorm(8, 128)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1) # → [B, 256, 8, 8]
        self.gn3   = nn.GroupNorm(8, 256)

        self.pool = nn.MaxPool2d(2, 2)

        # ---------- Convs after tiling (input: [B, 256 + 4, 8, 8]) ----------
        self.conv4 = nn.Conv2d(260, 256, 3, padding=1)
        self.gn4   = nn.GroupNorm(8, 256)
        
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.gn5   = nn.GroupNorm(8, 256)

        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.gn6   = nn.GroupNorm(8, 256)

        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.gn7   = nn.GroupNorm(8, 256)

        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.gn8   = nn.GroupNorm(8, 256)

        # ---------- Fully Connected Layers ----------
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_img, x_cond):
        """
        x_img:  [B, 1, 32, 32]
        x_cond: [B, 4]
        """

        # Initial conv + pooling stages
        x = self.pool(self.act(self.gn1(self.conv1(x_img))))   # → [B, 64, 16, 16]
        x = self.pool(self.act(self.gn2(self.conv2(x))))       # → [B, 128, 8, 8]
        x = self.act(self.gn3(self.conv3(x)))                  # → [B, 256, 8, 8]

        # Tile conditioning to 8x8
        cond = x_cond.unsqueeze(-1).unsqueeze(-1)              # [B, 4, 1, 1]
        cond_tiled = cond.expand(-1, -1, 8, 8)                 # [B, 4, 8, 8]

        # Concatenate along channel axis
        x = torch.cat([x, cond_tiled], dim=1)                  # [B, 260, 8, 8]

        # Deeper conv stack
        x = self.act(self.gn4(self.conv4(x)))                  # [B, 256, 8, 8]
        x = self.act(self.gn5(self.conv5(x)))                  # [B, 256, 8, 8]
        x = self.act(self.gn6(self.conv6(x)))                  # [B, 256, 8, 8]
        x = self.act(self.gn7(self.conv7(x)))                  # [B, 256, 8, 8]
        x = self.act(self.gn8(self.conv8(x)))                  # [B, 256, 8, 8]

        # Fully connected prediction head
        x = x.view(x.size(0), -1)                              # [B, 256*8*8 = 16384]
        x = self.dropout(self.act(self.fc1(x)))                # [B, 512]
        x = self.dropout(self.act(self.fc2(x)))                # [B, 128]
        out = self.fc3(x)                                      # [B, 1]

        return out
