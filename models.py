# For importing different models into ipynb
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

# DIFFUSION MODEL V1

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block, for image processing
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    """
    Downsampling path for U-Net, reduces spatial resolution while increasing feature depth
    Input: Image batch, size (batchsize, 1, 32, 32)
    Output: size (batchsize, out_channels, 16, 16)
    Output:
    """

    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(
            in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Doubles spatial dimensions, halves feature dimensions
        # My channel dimension for image will always be 1, greyscale
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        Doubles spatial size, but decreases channels:
        input: 2 vectors of size (binsize, in_channels / 2, h, w) 
        output: (binsize, outchannels, 2h, 2w)
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        """
        x is the upsampled features from previous decoder layer
        skip is the skip connection from the encoder, same size as x
        """
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    """
    Use FC layer for embedding 1-d metadata, like modes+weights
    (putting into higher dimension)
    Effectively our conditional
    input: Conditional, size (batchsize, input_dim = 4+4)
    Output: Higherdimensional tensor, size (batchsize, output_dim)

    """

    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    """
    U-Net style neural network for conditional image generation, conditioning on
    both timestep t and a context vector c (modes+weights)

    ** Diverges from MNIST Example:
        - instead of inputting n_classes, input the length of my
          conditional, as is continuous 8
    """

    def __init__(self, in_channels, n_feat=256, cond_dim=8):
        """
        in_channels (1 for greyscale), n_feat is base feature size, n_classes is number of labels
        """
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.cond_dim = cond_dim

        # Lifts image channels to n_feat using our residual block
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # (batchsize, 1, 32, 32) -> (batchsize, 2, 16, 16)
        self.down1 = UnetDown(n_feat, n_feat)
        # (batchsize, 2, 16, 16) -> (batchsize, 4, 8, 8)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        # reduces 2d feature map down2 into small latent vector of shape
        # (batchsize, 2*n_feat, 1, 1) via 7x7 average pooling
        self.to_vec = nn.Sequential(nn.AvgPool2d(8), nn.GELU())

        # Embeds timestep t and context c into vectors that will later be reshaped and added
        # to upsampling path
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(cond_dim, 2*n_feat)
        self.contextembed2 = EmbedFC(cond_dim, 1*n_feat)

        # upsamples latent vector (hiddenvec) back to 2d spatial mapping
        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            # otherwise just have 2*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        # upsampling (decoder) blocks that fuse upsampled features with skip connections from
        # the encoder
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # final processing layer to reduce features back to the original number of channels
        # Ideally produces the final denoised image!
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t):
        # x is (noisy) image, size (batchsize, in_channels, 32, 32)
        # c is context label,size (batchsize, 8)
        # t is timestep scalar, size (batchsize, 1)
        # Binary mask for whether to apply conditioning
        # probably will not need because need conditioning

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)  # pooled latent representation

        # convert context to one hot embedding

        # embed context, time step, reshapes them for broadcasting in upsampling layers
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


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

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / \
        T + beta1  # noise variance schedule (for every time t in T)
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
        # \bar{\alpha_t} , cumulative signal retention
        "alphabar_t": alphabar_t,
        # \sqrt{\bar{\alpha_t}} , scales clean image during noise
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}} , noise strength
        # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}} , for reverse diffusion
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model
    """

    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        """
        betas: tuple (beta1, beta2) for linear noise schedule
        n_T: total number of diffusion steps (e.g. 1000)
        drop_prob: probability of dropping conditioning (for classifier free guidance)
        """
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        x: clean image tensor, size (batchsize, 1, 32, 32)
        c: conditional vector, size (batchsize, 32)
        this method is used in training, so samples t and noise randomly
        """

        # t ~ Uniform(0, n_T)
        # sample a random timestep t for each item in the batch,
        # determines how much noise to add
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        # Generates noising image x_t from clean image x
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        # context_mask = torch.bernoulli(
        #     torch.zeros_like(c)+self.drop_prob).to(self.device)

        # return MSE between added noise, and our predicted noise
        # runs x_t, c, t, and context_mask through model, compares predicted noise
        # with actual noise using MSE
        # return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T))

    def sample(self, n_sample, size, device, c_i, guide_w=0.0):
        """
        n_sample: number of images to generate
        size: shape of each image, [1,32,32]
        guid_w: guidance strength, 0=no guidance, >0=stronger conditioning (what we want)
        """
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        # x_T ~ N(0, 1), sample initial noise
        x_i = torch.randn(n_sample, *size).to(device)  # start from pure noise
        # context for us just cycles throught the mnist labels

        x_i_store = []  # keep track of generated steps in case want to plot something
        print()
        # Iterate over timesteps in revers (from noise -> image)
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor(
                [i / self.n_T], device=device).repeat(n_sample, 1)

            # add noise at all steps except final one
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # predict the noise using both conditioned and unconditioned branches
            eps = self.nn_model(x_i, c_i, t_is)
            # apply classifier-free guidance formula: ϵ = (1 + w)⋅ϵ_cond − w⋅ϵ_uncond

            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            # save frames for visualization every 20 steps and near the end
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        # returns final denoised image x_i and intermediate steps x_i_store
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

# CNN MODEL V1