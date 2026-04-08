import numpy as np
import torch
import torch.nn as nn


class EmbedFC(nn.Module):
    """
    Feedforward embedding network used to map low-dimensional inputs
    (e.g., scalar timesteps) into a higher-dimensional conditioning space.
    """

    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CondSeqEncoder(nn.Module):
    """
    Encoder for ordered conditioning sequences.

    Takes a sequence of conditioning tokens (e.g., wavelength-dependent
    parameters, modes, and weights) and produces a fixed-size embedding.

    Architecture:
        - Linear projection into model dimension
        - Learned positional encoding
        - Transformer encoder layers
        - Mean pooling across sequence dimension
        - Final projection to conditioning vector

    Input:
        c_seq : (B, K, F)

    Output:
        u     : (B, u_dim)
    """

    def __init__(self, feat_dim=12, u_dim=128, d_model=128, nhead=4, num_layers=2, max_len=24, dropout=0.0):
        super().__init__()

        self.in_proj = nn.Linear(feat_dim, d_model)

        # Learned positional encoding to preserve ordering of sequence elements
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, u_dim)

    def forward(self, c_seq):
        h = self.in_proj(c_seq)
        h = h + self.pos[:, :h.size(1), :]
        h = self.encoder(h)

        # Aggregate sequence information into a global representation
        h = h.mean(dim=1)

        return self.out_proj(h)


class FiLMConvBlock(nn.Module):
    """
    Convolutional block with FiLM conditioning and normalization.

    Structure:
        Conv2d → GroupNorm → FiLM modulation → GELU

    FiLM parameters are generated from a conditioning vector and applied
    channel-wise to the normalized activations.
    """

    def __init__(self, in_ch, out_ch, u_dim, stride=1, groups=8, hidden=128):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch, out_ch, 3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.gn = nn.GroupNorm(groups, out_ch)
        self.act = nn.GELU()

        # MLP mapping conditioning vector → (gamma, beta)
        self.to_gb = nn.Sequential(
            nn.Linear(u_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * out_ch),
        )

    def forward(self, x, u):
        h = self.conv(x)
        h = self.gn(h)

        g, b = self.to_gb(u).chunk(2, dim=-1)

        # Apply channel-wise affine modulation
        h = (1.0 + g[..., None, None]) * h + b[..., None, None]

        return self.act(h)


class UnetDownFiLM(nn.Module):
    """
    Downsampling block for U-Net with FiLM conditioning.

    Applies two FiLMConvBlocks, where the second block performs
    spatial downsampling via stride-2 convolution.
    """

    def __init__(self, in_ch, out_ch, u_dim, groups=8):
        super().__init__()

        self.b1 = FiLMConvBlock(in_ch, out_ch, u_dim, stride=1, groups=groups)
        self.b2 = FiLMConvBlock(out_ch, out_ch, u_dim, stride=2, groups=groups)

    def forward(self, x, u):
        x = self.b1(x, u)
        x = self.b2(x, u)
        return x


class UnetUpFiLM(nn.Module):
    """
    Upsampling block for U-Net with FiLM conditioning.

    Structure:
        - Transposed convolution for upsampling
        - Skip connection concatenation
        - Two FiLMConvBlocks for feature refinement
    """

    def __init__(self, in_ch, skip_ch, out_ch, u_dim, groups=8):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)

        self.b1 = FiLMConvBlock(out_ch + skip_ch, out_ch,
                                u_dim, stride=1, groups=groups)
        self.b2 = FiLMConvBlock(out_ch, out_ch, u_dim, stride=1, groups=groups)

    def forward(self, x, skip, u):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.b1(x, u)
        x = self.b2(x, u)
        return x


class ContextUnetFiLM(nn.Module):
    """
    FiLM-conditioned U-Net used as the denoising backbone in diffusion.

    The model predicts the noise component given:
        - noisy input image x_t
        - conditioning sequence c_seq
        - timestep t

    Conditioning is incorporated via:
        - Transformer-based sequence encoder for c_seq
        - MLP embedding for timestep t
        - FiLM modulation applied throughout the network
    """

    def __init__(self, in_channels=1, base=64, cond_feat_dim=12, K=24, u_dim=128, groups=8,
                 d_model=128, nhead=4, n_layers=2):
        super().__init__()

        self.in_channels = in_channels

        # Encode conditioning sequence into global embedding
        self.c_embed = CondSeqEncoder(
            feat_dim=cond_feat_dim,
            u_dim=u_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=n_layers,
            max_len=K,
            dropout=0.0,
        )

        # Embed timestep into same conditioning space
        self.t_embed = EmbedFC(1, u_dim, hidden=2 * u_dim)

        # Initial feature extraction
        self.init1 = FiLMConvBlock(
            in_channels, base, u_dim, stride=1, groups=groups)
        self.init2 = FiLMConvBlock(base, base, u_dim, stride=1, groups=groups)

        # Encoder path
        self.down1 = UnetDownFiLM(base, base, u_dim, groups=groups)
        self.down2 = UnetDownFiLM(base, 2 * base, u_dim, groups=groups)
        self.down3 = UnetDownFiLM(2 * base, 4 * base, u_dim, groups=groups)

        # Bottleneck
        self.mid1 = FiLMConvBlock(
            4 * base, 4 * base, u_dim, stride=1, groups=groups)
        self.mid2 = FiLMConvBlock(
            4 * base, 4 * base, u_dim, stride=1, groups=groups)

        # Decoder path with skip connections
        self.up2 = UnetUpFiLM(4 * base, 2 * base, 2 *
                              base, u_dim, groups=groups)
        self.up1 = UnetUpFiLM(2 * base, base, base, u_dim, groups=groups)
        self.up0 = UnetUpFiLM(base, base, base, u_dim, groups=groups)

        # Output layer predicting noise
        self.out = nn.Conv2d(base, in_channels, 3, 1, 1)

    def forward(self, x, c_seq, t):
        # Combine conditioning sources into a single embedding
        u = self.c_embed(c_seq) + self.t_embed(t)

        x0 = self.init1(x, u)
        x0 = self.init2(x0, u)

        d1 = self.down1(x0, u)
        d2 = self.down2(d1, u)
        d3 = self.down3(d2, u)

        m = self.mid1(d3, u)
        m = self.mid2(m, u)

        u2 = self.up2(m, d2, u)
        u1 = self.up1(u2, d1, u)
        u0 = self.up0(u1, x0, u)

        return self.out(u0)


def ddpm_schedules(beta1: float, beta2: float, T: int):
    """
    Precompute diffusion schedules used in DDPM.

    Returns tensors for forward and reverse diffusion steps, including:
        - beta_t, alpha_t
        - cumulative products (alphabar)
        - square roots and derived coefficients
    """

    assert 0.0 < beta1 < beta2 < 1.0
    assert T >= 1

    beta_t = torch.linspace(beta1, beta2, T, dtype=torch.float32)
    beta_t = torch.cat([torch.zeros(1, dtype=torch.float32), beta_t], 0)

    alpha_t = 1.0 - beta_t
    alpha_t[0] = 1.0

    alphabar_t = torch.cumprod(alpha_t, dim=0)
    alphabar_t[0] = 1.0

    sqrt_beta_t = torch.sqrt(beta_t)
    sqrtab = torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(1.0 - alphabar_t)

    oneover_sqrta = 1.0 / torch.sqrt(alpha_t)
    beta_over_sqrtmab = beta_t / torch.clamp(sqrtmab, min=1e-20)

    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "beta_over_sqrtmab": beta_over_sqrtmab
    }


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) wrapper.

    Trains a neural network to predict the noise added to data during
    the forward diffusion process, and uses this model to generate samples
    via iterative denoising.
    """

    def __init__(self, nn_model, betas, n_T, device):
        super().__init__()

        self.nn_model = nn_model.to(device)

        # Register diffusion schedule tensors as buffers
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()

    def forward(self, x0, c_seq):
        # Sample random timestep for each element in batch
        B = x0.size(0)
        t_int = torch.randint(1, self.n_T + 1, (B,), device=self.device)

        # Sample Gaussian noise
        eps = torch.randn_like(x0)

        # Forward diffusion: construct noisy input x_t
        x_t = self.sqrtab[t_int, None, None, None] * x0 + \
            self.sqrtmab[t_int, None, None, None] * eps

        t = (t_int / self.n_T).float().unsqueeze(1)

        # Predict noise and compute MSE loss
        eps_hat = self.nn_model(x_t, c_seq, t)
        return self.loss_mse(eps_hat, eps)

    @torch.no_grad()
    def sample(self, n_sample, size, device, c_seq):
        # Initialize with Gaussian noise
        x = torch.randn(n_sample, *size, device=device)
        x_store = []

        self.nn_model.eval()

        # Reverse diffusion process
        for i in range(self.n_T, 0, -1):
            t = torch.full((n_sample, 1), i / self.n_T, device=device)

            z = torch.randn(n_sample, *size, device=device) if i > 1 else 0.0

            eps_hat = self.nn_model(x, c_seq, t)

            x = self.oneover_sqrta[i] * (
                x - eps_hat * self.beta_over_sqrtmab[i]
            ) + self.sqrt_beta_t[i] * z

            # Store intermediate states for visualization
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_store.append(x.detach().cpu().numpy())

        return x, np.array(x_store)


if __name__ == "__main__":
    # Minimal sanity check for forward and sampling passes

    model = ContextUnetFiLM(
        in_channels=1,
        base=64,
        cond_feat_dim=12,
        K=24,
        u_dim=128,
        groups=8,
        d_model=128,
        nhead=4,
        n_layers=2
    )

    ddpm = DDPM(model, betas=(1e-4, 0.02), n_T=10, device="cpu")

    x0 = torch.zeros(4, 1, 32, 32)
    c_seq = torch.zeros(4, 24, 12)

    loss = ddpm(x0, c_seq)
    print("loss:", loss.item())

    samples, history = ddpm.sample(
        n_sample=4,
        size=(1, 32, 32),
        device="cpu",
        c_seq=c_seq
    )

    print("samples:", samples.shape, "history:", history.shape)
