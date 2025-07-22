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
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CondEncoderWeighted(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU()
        )
        self.attn = nn.Sequential(
            nn.Linear(out_dim, 1)  # attention score for each pair
        )

    def forward(self, x):
        """
        x: [B, 4, 2] - 4 (mode, weight) pairs per sample
        returns: [B, out_dim] - pooled representation of pairs
        """
        h = self.encoder(x)                      # [B, 4, out_dim]
        attn_scores = self.attn(h).squeeze(-1)   # [B, 4]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, 4]
        attn_weights = attn_weights.unsqueeze(-1)     # [B, 4, 1]
        pooled = (h * attn_weights).sum(dim=1)        # [B, out_dim]
        return pooled
    

class Net4_Mode0Weight_4x2(nn.Module):
    """
    Predicts only the 0th mode-weight pair [mode0, weight0] using
    structured conditional input [B, 4, 2] and a single output head [B, 2].
    Uses GroupNorm for conv layers and LayerNorm for FC layers.
    """
    def __init__(self):
        super().__init__()

        # ---------- CNN trunk with GroupNorm ----------
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 1, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, 1, 1), nn.GroupNorm(16, 256), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, 1, 1), nn.GroupNorm(32, 512), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            Flatten()  # [B, 8192]
        )

        # ---------- Conditional encoder ----------
        self.cond_encoder = CondEncoderWeighted(in_dim=2, hidden_dim=32, out_dim=128)

        # ---------- Fully-connected trunk with LayerNorm ----------
        self.fc = nn.Sequential(
            nn.Linear(8192 + 128, 2048), nn.LayerNorm(2048), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(2048, 1024),       nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 256),        nn.LayerNorm(256),  nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 2)  # Joint prediction: [mode0, weight0]
        )

    def forward(self, x_img, x_cond):
        """
        x_img: [B, 1, 32, 32]
        x_cond: [B, 4, 2] - 4 mode-weight pairs
        """
        img_feat = self.cnn(x_img)                 # [B, 8192]
        cond_feat = self.cond_encoder(x_cond)      # [B, 128]
        x = torch.cat((img_feat, cond_feat), dim=1)  # [B, 8320]
        return self.fc(x)                          # [B, 2]
