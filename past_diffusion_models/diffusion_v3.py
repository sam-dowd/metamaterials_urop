from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from torch import autograd
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.optim as optim
import torchvision.datasets as datasets
import time
import os

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, C, -1)
        k = self.k(x).reshape(B, C, -1)
        v = self.v(x).reshape(B, C, -1)
        attn = torch.softmax(q.transpose(1,2) @ k / (C**0.5), dim=-1)
        out = (attn @ v.transpose(1,2)).transpose(1,2).reshape(B, C, H, W)
        return self.proj(out) + x
    
class AdaIN(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.fc = nn.Linear(cond_dim, channels*2)
    def forward(self, x, cond):
        h = self.fc(cond)
        gamma, beta = h.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        mean = x.mean([2,3], keepdim=True)
        std = x.std([2,3], keepdim=True)
        x_norm = (x - mean) / (std + 1e-5)
        return gamma * x_norm + beta
    

def get_timestep_embedding(timesteps, embedding_dim):
    """
    timesteps: 1-D  (B,)  or 2-D (B,1) tensor of integers / floats
    returns:   (B, embedding_dim) sinusoidal embedding
    """
    if timesteps.ndim == 2:
        timesteps = timesteps.squeeze(-1)          # (B,)
    assert timesteps.ndim == 1                     # ensure 1-D
    half_dim = embedding_dim // 2
    exponents = torch.arange(half_dim, device=timesteps.device) / half_dim
    freqs = 10000 ** (-exponents)
    angles = timesteps.float()[:, None] * freqs[None, :]  # (B, half_dim)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb                                    # (B, embedding_dim)

class ImprovedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=8, use_attention=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.ada = AdaIN(out_channels, cond_dim)
        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention(out_channels)
        else:
            self.attn = nn.Identity()
    def forward(self, x, cond):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.ada(h, cond)
        h = F.gelu(self.norm2(self.conv2(h)))
        h = self.attn(h)
        if self.same_channels:
            return (x + h) / 1.414
        else:
            return h
        
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=1, base=64, cond_dim=8, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = nn.Linear(time_dim, cond_dim)
        # Down
        self.enc1 = ImprovedResBlock(in_channels, base, cond_dim, use_attention=False)
        self.enc2 = ImprovedResBlock(base, base*2, cond_dim, use_attention=True)
        self.enc3 = ImprovedResBlock(base*2, base*4, cond_dim, use_attention=True)
        self.enc4 = ImprovedResBlock(base*4, base*8, cond_dim, use_attention=True)
        # Up
        self.up1 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec1 = ImprovedResBlock(base*8, base*4, cond_dim, use_attention=True)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = ImprovedResBlock(base*4, base*2, cond_dim, use_attention=True)
        self.up3 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec3 = ImprovedResBlock(base*2, base, cond_dim, use_attention=False)
        self.out = nn.Conv2d(base, in_channels, 1)
    def forward(self, x, cond, t, context_mask=0):
        cond = cond * (1-context_mask)
        t_emb = get_timestep_embedding(t, self.time_dim).to(x.device)
        cond = cond + self.time_embed(t_emb)
        e1 = self.enc1(x, cond)
        e2 = self.enc2(F.avg_pool2d(e1, 2), cond)
        e3 = self.enc3(F.avg_pool2d(e2, 2), cond)
        e4 = self.enc4(F.avg_pool2d(e3, 2), cond)
        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], 1)
        d1 = self.dec1(d1, cond)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.dec2(d2, cond)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], 1)
        d3 = self.dec3(d3, cond)
        return self.out(d3)
    
def ddpm_schedules(beta1, beta2, T):
    """
    Precomputes all noise scheduling terms needed for training and sampling
    from a denoising diffusion probabilistic model
    Uses a sequence of gradually increasing noise level over T timesteps
    beta1: starting noise level, O(1e-4)
    beta2: final noise level, O(0.02)
    T: number of time steps
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1 # noise variance schedule (for every time t in T)
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    # dictionary of schedule terms
    return {
        "alpha_t": alpha_t,  # \alpha_t , signal retention at time step t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t} , noise scaling factor
        "alphabar_t": alphabar_t,  # \bar{\alpha_t} , cumulative signal retention
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}} , scales clean image during noise
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}} , noise strength
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}} , for reverse diffusion
    }


class improved_DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model with Classifier-Free Guidance
    --------------------------------------------------------------------
    * `nn_model(x, c, t, context_mask)` must accept the extra boolean mask.
      When mask==1 the model should ignore / zero the conditioning.
    """
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super().__init__()
        self.nn_model  = nn_model.to(device)
        for k, v in ddpm_schedules(*betas, n_T).items():
            self.register_buffer(k, v)

        self.n_T      = n_T
        self.device   = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        x : [B, 1, 32, 32] clean image
        c : [B, cond_dim]  conditioning vector
        """
        B = x.size(0)
        t  = torch.randint(1, self.n_T + 1, (B,), device=self.device)
        eps = torch.randn_like(x)          # noise ~ N(0,1)

        x_t = self.sqrtab[t, None, None, None] * x \
            + self.sqrtmab[t, None, None, None] * eps

        # sample mask: 1 → DROP conditioning, 0 → keep
        context_mask = torch.bernoulli(
            torch.full((B, 1), self.drop_prob, device=self.device)
        )

        eps_pred = self.nn_model(x_t, c, t / self.n_T, context_mask)
        return self.loss_mse(eps, eps_pred)

    @torch.no_grad()
    def sample(self, n_sample, size, device, c_i, guide_w=0.0):
        """
        guide_w = 0   → unconditional
        guide_w = 1   → full cond − uncond blend (as in paper)
        guide_w > 1   → stronger conditioning
        """
        x = torch.randn(n_sample, *size, device=device)  # x_T
        store = []

        for i in range(self.n_T, 0, -1):
            t = torch.full((n_sample, 1), i / self.n_T, device=device)

            # ---------- predict noise with and without context ----------
            eps_cond  = self.nn_model(x,  c_i, t, torch.zeros_like(t))  # mask=0
            eps_uncond= self.nn_model(x,  torch.zeros_like(c_i), t,
                                      torch.ones_like(t))              # mask=1

            # guidance:  ε = ε_u  + w (ε_c - ε_u)
            eps = eps_uncond + guide_w * (eps_cond - eps_uncond)

            z = torch.randn_like(x) if i > 1 else 0
            x = ( self.oneover_sqrta[i] *
                  (x - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z )

            if i % 20 == 0 or i == self.n_T or i < 8:
                store.append(x.cpu().numpy())

        return x, np.array(store)

class CondSequential(nn.Module):
    """Sequential that passes (x, cond) to each sub-module."""
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, cond):
        for layer in self.layers:
            x = layer(x, cond)
        return x
    
class ImprovedUNet2(nn.Module):
    def __init__(self, in_channels=1, base=128, cond_dim=8, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, cond_dim)
        )

        # ─── Encoder ───────────────────────────────────────────────
        self.enc1 = CondSequential(
            ImprovedResBlock(in_channels, base, cond_dim),
            ImprovedResBlock(base, base, cond_dim)
        )
        self.enc2 = CondSequential(
            ImprovedResBlock(base, base*2, cond_dim, use_attention=True),
            ImprovedResBlock(base*2, base*2, cond_dim, use_attention=True)
        )
        self.enc3 = CondSequential(
            ImprovedResBlock(base*2, base*4, cond_dim, use_attention=True),
            ImprovedResBlock(base*4, base*4, cond_dim, use_attention=True)
        )
        self.enc4 = CondSequential(
            ImprovedResBlock(base*4, base*8, cond_dim, use_attention=True),
            ImprovedResBlock(base*8, base*8, cond_dim, use_attention=True)
        )

        # ─── Bottleneck ────────────────────────────────────────────
        self.mid = CondSequential(
            ImprovedResBlock(base*8, base*8, cond_dim, use_attention=True),
            ImprovedResBlock(base*8, base*8, cond_dim, use_attention=True)
        )

        # ─── Decoder ───────────────────────────────────────────────
        self.up1 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec1 = CondSequential(
            ImprovedResBlock(base*8, base*4, cond_dim, use_attention=True),
            ImprovedResBlock(base*4, base*4, cond_dim, use_attention=True)
        )

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = CondSequential(
            ImprovedResBlock(base*4, base*2, cond_dim, use_attention=True),
            ImprovedResBlock(base*2, base*2, cond_dim, use_attention=True)
        )

        self.up3 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec3 = CondSequential(
            ImprovedResBlock(base*2, base, cond_dim),
            ImprovedResBlock(base, base, cond_dim)
        )

        self.out = nn.Conv2d(base, in_channels, 1)

    def forward(self, x, cond, t, context_mask=0):
        cond = cond * (1 - context_mask)
        t_emb = get_timestep_embedding(t, self.time_dim).to(x.device)
        cond = cond + self.time_embed(t_emb)

        # Encoder
        e1 = self.enc1(x, cond)
        e2 = self.enc2(F.avg_pool2d(e1, 2), cond)
        e3 = self.enc3(F.avg_pool2d(e2, 2), cond)
        e4 = self.enc4(F.avg_pool2d(e3, 2), cond)

        # Bottleneck
        h = self.mid(e4, cond)

        # Decoder
        d1 = self.up1(h)
        d1 = self.dec1(torch.cat([d1, e3], dim=1), cond)

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), cond)

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, e1], dim=1), cond)

        return self.out(d3)