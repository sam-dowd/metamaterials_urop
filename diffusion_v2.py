import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from skimage.metrics import structural_similarity as ssim
from scipy import linalg
from tqdm import tqdm


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
        attn = torch.softmax(q.transpose(1, 2) @ k / (C**0.5), dim=-1)
        out = (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
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
        mean = x.mean([2, 3], keepdim=True)
        std = x.std([2, 3], keepdim=True)
        x_norm = (x - mean) / (std + 1e-5)
        return gamma * x_norm + beta


def get_timestep_embedding(timesteps, embedding_dim):
    # Sinusoidal positional encoding
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,
                    device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class ImprovedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, use_attention=False):
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
        self.enc1 = ImprovedResBlock(
            in_channels, base, cond_dim, use_attention=False)
        self.enc2 = ImprovedResBlock(
            base, base*2, cond_dim, use_attention=True)
        self.enc3 = ImprovedResBlock(
            base*2, base*4, cond_dim, use_attention=True)
        self.enc4 = ImprovedResBlock(
            base*4, base*8, cond_dim, use_attention=True)
        # Up
        self.up1 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec1 = ImprovedResBlock(
            base*8, base*4, cond_dim, use_attention=True)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = ImprovedResBlock(
            base*4, base*2, cond_dim, use_attention=True)
        self.up3 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec3 = ImprovedResBlock(
            base*2, base, cond_dim, use_attention=False)
        self.out = nn.Conv2d(base, in_channels, 1)

    def forward(self, x, cond, t):
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


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def ddim_sample(model, x, cond, n_steps, eta=0.0):
    # x: initial noise, cond: conditioning, n_steps: number of steps
    # eta=0.0 is deterministic, >0 adds noise
    device = x.device
    betas = cosine_beta_schedule(n_steps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    x_t = x
    for i in reversed(range(n_steps)):
        t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
        eps = model(x_t, cond, t.float() / n_steps)
        if i == 0:
            x_t = (
                x_t - sqrt_one_minus_alphas_cumprod[i] * eps) / sqrt_alphas_cumprod[i]
        else:
            x_prev = (
                x_t - sqrt_one_minus_alphas_cumprod[i] * eps) / sqrt_alphas_cumprod[i]
            noise = torch.randn_like(x_t) if eta > 0 else 0
            x_t = sqrt_alphas_cumprod[i-1] * x_prev + \
                sqrt_one_minus_alphas_cumprod[i-1] * noise
    return x_t


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        # x, y: (B, 1, H, W) -> (B, 3, H, W)
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        fx = self.vgg(x)
        fy = self.vgg(y)
        return F.l1_loss(fx, fy)


def augment(x):
    # x: (B, 1, H, W)
    if torch.rand(1) < 0.5:
        x = TF.hflip(x)
    if torch.rand(1) < 0.5:
        x = TF.vflip(x)
    if torch.rand(1) < 0.5:
        x = TF.rotate(x, angle=torch.randint(-30, 30, (1,)).item())
    return x


def compute_ssim(img1, img2):
    # img1, img2: (H, W) numpy arrays
    return ssim(img1, img2, data_range=img2.max() - img2.min())


def compute_fid(real_acts, fake_acts):
    # real_acts, fake_acts: (N, D)
    mu1, sigma1 = real_acts.mean(0), np.cov(real_acts, rowvar=False)
    mu2, sigma2 = fake_acts.mean(0), np.cov(fake_acts, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)


def split_dataset(dataset, train_ratio=0.8, random_seed=42):
    """
    Split dataset into train and test sets
    """
    import random
    from torch.utils.data import Subset

    # Set random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Get total dataset size
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    # Create indices and shuffle
    indices = list(range(total_size))
    random.shuffle(indices)

    # Split indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    return train_dataset, test_dataset
