import torch
import torch.nn as nn


class FiLMConvBlock(nn.Module):
    """
    Convolutional block with FiLM conditioning.

    Each block applies:
        Conv2d → FiLM modulation → GELU activation

    Conditioning is implemented via Feature-wise Linear Modulation (FiLM),
    where channel-wise affine parameters (gamma, beta) are generated from
    a conditioning vector and applied as:

        h = (1 + gamma) * h + beta

    This allows the network to adapt intermediate feature representations
    based on external inputs (e.g., wavelength or physical parameters).
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, hidden: int = 64):
        super().__init__()

        # Spatial feature extraction via convolution (no bias due to FiLM affine shift)
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Nonlinear activation applied after FiLM modulation
        self.act = nn.GELU()

        # MLP mapping conditioning vector → per-channel (gamma, beta)
        self.film = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * out_ch),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Apply convolution to input feature map
        h = self.conv(x)

        # Generate FiLM parameters from conditioning input
        gb = self.film(cond)
        gamma, beta = gb.chunk(2, dim=-1)

        # Reshape for channel-wise broadcasting across spatial dimensions
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        # Apply FiLM modulation followed by activation
        h = (1.0 + gamma) * h + beta
        h = self.act(h)

        return h


class FilmCNN(nn.Module):
    """
    FiLM-conditioned convolutional neural network for waveguide regression.

    The model processes a 2D waveguide pattern and conditions all intermediate
    feature maps on an external vector (e.g., wavelength or physical parameters)
    using FiLM at every convolutional block.

    Architecture:
        - Stacked FiLMConvBlocks with increasing channel width
        - Spatial downsampling via MaxPool2d (factor of 2 at each stage)
        - Global feature aggregation via AdaptiveAvgPool2d
        - Fully connected regression head

    The network outputs a scalar prediction per input sample.
    """

    def __init__(self, cond_dim=1, p_drop=0.1, film_hidden=64):
        super().__init__()

        # Initial feature extraction at full resolution (32×32)
        self.b1 = FiLMConvBlock(1, 32, cond_dim, hidden=film_hidden)
        self.b2 = FiLMConvBlock(32, 32, cond_dim, hidden=film_hidden)

        # Intermediate representation at reduced resolution (16×16)
        self.b3 = FiLMConvBlock(32, 64, cond_dim, hidden=film_hidden)
        self.b4 = FiLMConvBlock(64, 64, cond_dim, hidden=film_hidden)

        # Deeper feature representation (8×8)
        self.b5 = FiLMConvBlock(64, 128, cond_dim, hidden=film_hidden)
        self.b6 = FiLMConvBlock(128, 128, cond_dim, hidden=film_hidden)

        # High-level representation (4×4)
        self.b7 = FiLMConvBlock(128, 128, cond_dim, hidden=film_hidden)
        self.b8 = FiLMConvBlock(128, 128, cond_dim, hidden=film_hidden)
        self.b9 = FiLMConvBlock(128, 128, cond_dim, hidden=film_hidden)
        self.b10 = FiLMConvBlock(128, 128, cond_dim, hidden=film_hidden)

        # Spatial downsampling operator (reduces H, W by factor of 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global pooling to produce a single feature vector per sample
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected regression head
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(64, 1),
        )

        # Weight initialization consistent with standard CNN/MLP practices
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img32x32: torch.Tensor, cond2: torch.Tensor):
        """
        Forward pass.

        Args:
            img32x32 : input waveguide pattern, shape (B, 1, 32, 32)
            cond2    : conditioning vector, shape (B, cond_dim)

        Returns:
            y : scalar prediction per sample, shape (B,)
        """

        # Stage 1: feature extraction at full resolution
        x = self.b1(img32x32, cond2)
        x = self.b2(x, cond2)

        # Downsample: 32 → 16
        x = self.pool2(x)

        # Stage 2: intermediate features
        x = self.b3(x, cond2)
        x = self.b4(x, cond2)

        # Downsample: 16 → 8
        x = self.pool2(x)

        # Stage 3: deeper features
        x = self.b5(x, cond2)
        x = self.b6(x, cond2)

        # Downsample: 8 → 4
        x = self.pool2(x)

        # Stage 4: high-level representation
        x = self.b7(x, cond2)
        x = self.b8(x, cond2)
        x = self.b9(x, cond2)
        x = self.b10(x, cond2)

        # Global pooling and flattening
        f = self.pool(x).squeeze(-1).squeeze(-1)

        # Final regression output
        y = self.head(f).squeeze(-1)

        return y
