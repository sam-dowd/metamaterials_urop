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
    """
    Flatten function so can include in nn.Sequential(...)
    """

    def forward(self, x):
        return torch.flatten(x, 1)  # flatten all dimensions except batch


class Net4(nn.Module):
    """
    Deeper and wider CNN + FC architecture for regression on waveguide input with conditional features.
    """

    def __init__(self):
        super(Net4, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1,
                      padding=1),    # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1,
                      padding=1),  # [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.GELU(),
            # [B, 128, 16, 16]
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, stride=1,
                      padding=1),  # [B, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.GELU(),
            # [B, 256, 8, 8]
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, kernel_size=3, stride=1,
                      padding=1),  # [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.GELU(),
            # [B, 512, 4, 4]
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # [B, 512 * 4 * 4 = 8192]
            Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(8192 + 4, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(256, 8)  # Output
        )

    def forward(self, x_img, x_cond):
        x = self.cnn(x_img)                 # [B, 8192]
        x = torch.cat((x, x_cond), dim=1)   # [B, 8196]
        return self.fc(x)
