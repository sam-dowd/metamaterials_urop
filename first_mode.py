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
    
class Net4_Mode0Weight0(nn.Module):
    """
    Net4 variant that predicts only:
    - mode 0  (original index 0)
    - weight0 (original index 4)
    Output shape: [B, 2]
    """
    def __init__(self):
        super().__init__()

        # ---------- CNN trunk (unchanged) ----------
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            Flatten()                               # -> [B, 8192]
        )

        # ---------- Fully-connected head ----------
        self.fc = nn.Sequential(
            nn.Linear(8192 + 4, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(2048, 1024),     nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 256),      nn.BatchNorm1d(256),  nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 2)                           # <- predict [mode0, weight0]
        )

    def forward(self, x_img, x_cond):
        x = self.cnn(x_img)                   # [B, 8192]
        x = torch.cat((x, x_cond), dim=1)     # [B, 8196]
        return self.fc(x)                     # [B, 2]

class Net4_Mode0Weight0_two_head(nn.Module):
    """
    Net4 variant that predicts:
    - mode 0 (original index 0)
    - weight 0 (original index 4)
    Output shape: [B, 2]
    Uses two separate output heads.
    """
    def __init__(self):
        super().__init__()

        # ---------- CNN trunk ----------
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            Flatten()  # -> [B, 8192]
        )

        # ---------- Shared FC trunk ----------
        self.shared = nn.Sequential(
            nn.Linear(8192 + 4, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(2048, 1024),     nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 256),      nn.BatchNorm1d(256),  nn.GELU(), nn.Dropout(0.25),
        )

        # ---------- Split output heads ----------
        self.head_mode0 = nn.Linear(256, 1)
        self.head_weight0 = nn.Linear(256, 1)

    def forward(self, x_img, x_cond):
        x = self.cnn(x_img)                    # [B, 8192]
        x = torch.cat((x, x_cond), dim=1)      # [B, 8196]
        x = self.shared(x)                     # [B, 256]

        mode0   = self.head_mode0(x)           # [B, 1]
        weight0 = self.head_weight0(x)         # [B, 1]

        return torch.cat([mode0, weight0], dim=1)  # [B, 2]
