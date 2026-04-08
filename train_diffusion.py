from diffusion_dispersion import ContextUnetFiLM, DDPM
# Diffusion backbone and training wrapper for conditional waveguide generation

from h5_dataset_diff import H5WaveguideDataset
# Dataset providing waveguide patterns together with dispersion-based conditioning data

import matplotlib.pyplot as plt
import math
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
# Non-interactive backend for file-based plotting in headless environments


# ---------------- Configuration ----------------
EPOCHS = 240
BATCH_SIZE = 256
VAL_FRAC = 0.3
LR = 3e-4
NUM_WORKERS = 4
PIN_MEMORY = True
SEED = 42

N_T = 1000
BETAS = (1e-4, 0.02)
BASE = 64
U_DIM = 128
GROUPS = 8

K = 24
COND_FEAT_DIM = 12

OUT_DIR = Path("diffusion_dispersion_model")
# ----------------------------------------------


def set_seed(seed=42):
    # Fix all stochastic sources to promote reproducibility across runs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pack_cond(params, modes, weights):
    # Assemble the per-token conditioning representation by concatenating
    # geometric/physical parameters, modal values, and modal weights
    return torch.cat([params, modes, weights], dim=-1)


@torch.no_grad()
def eval_epoch(ddpm: DDPM, loader: DataLoader, device: torch.device) -> float:
    # Estimate the mean validation objective over one full pass through the loader
    ddpm.eval()
    total, count = 0.0, 0

    for x0, params, modes, weights in loader:
        x0 = x0.float().to(device, non_blocking=True)
        c_seq = pack_cond(params, modes, weights).float().to(
            device, non_blocking=True
        )

        loss = ddpm(x0, c_seq)

        bs = x0.size(0)
        total += loss.item() * bs
        count += bs

    return total / max(1, count)


@torch.no_grad()
def make_val_grid(ddpm: DDPM, val_ds: Subset, device: torch.device, out_path: Path):
    # Generate a qualitative comparison grid between validation targets
    # and samples produced under the same conditioning inputs
    ddpm.eval()

    n_pairs = 18
    idxs = torch.randperm(len(val_ds))[:n_pairs].tolist()

    x0_list, c_list = [], []

    for j in idxs:
        x0, params, modes, weights = val_ds[j]
        x0 = x0.unsqueeze(0)
        c_seq = pack_cond(
            params.unsqueeze(0),
            modes.unsqueeze(0),
            weights.unsqueeze(0),
        )
        x0_list.append(x0)
        c_list.append(c_seq)

    x0_batch = torch.cat(x0_list, dim=0).float().to(device)
    c_batch = torch.cat(c_list, dim=0).float().to(device)

    x_gen, _ = ddpm.sample(
        n_sample=n_pairs,
        size=(1, 32, 32),
        device=device,
        c_seq=c_batch,
    )

    fig, axes = plt.subplots(6, 6, figsize=(12, 12))

    for k in range(n_pairs):
        r = k // 3
        pair_in_row = k % 3
        c_t = 2 * pair_in_row
        c_g = c_t + 1

        tgt = x0_batch[k, 0].detach().cpu().numpy()
        gen = x_gen[k, 0].detach().cpu().numpy()

        axes[r, c_t].imshow(tgt, cmap="Reds", origin="upper")
        axes[r, c_g].imshow(gen, cmap="Greys", origin="upper")
        axes[r, c_t].set_axis_off()
        axes[r, c_g].set_axis_off()

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    # Initialize reproducibility, device placement, and output directory
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- Data split ----------------
    # Construct a deterministic train/validation partition from the training set
    full_train = H5WaveguideDataset(split="train")
    n = len(full_train)

    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(n, generator=g)

    n_val = math.ceil(VAL_FRAC * n)
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_train, val_idx)

    # Data loaders for stochastic optimization and held-out evaluation
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    # ---------------- Model ----------------
    # Instantiate the FiLM-conditioned U-Net denoiser and wrap it
    # in the DDPM training/sampling interface
    unet = ContextUnetFiLM(
        in_channels=1,
        base=BASE,
        cond_feat_dim=COND_FEAT_DIM,
        K=K,
        u_dim=U_DIM,
        groups=GROUPS,
    )
    ddpm = DDPM(nn_model=unet, betas=BETAS, n_T=N_T, device=device).to(device)

    # AdamW optimizer with cosine annealing over the full training horizon
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.01,
    )

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_path = OUT_DIR / "best_model.pth"
    last_path = OUT_DIR / "last_model.pth"

    # ---------------- Training loop ----------------
    # Optimize the diffusion objective and track the best validation checkpoint
    for epoch in range(1, EPOCHS + 1):
        ddpm.train()
        running, count = 0.0, 0

        for x0, params, modes, weights in train_loader:
            x0 = x0.float().to(device, non_blocking=True)
            c_seq = pack_cond(params, modes, weights).float().to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)
            loss = ddpm(x0, c_seq)
            loss.backward()
            optimizer.step()

            bs = x0.size(0)
            running += loss.item() * bs
            count += bs

        train_loss = running / max(1, count)
        val_loss = eval_epoch(ddpm, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ddpm.state_dict(), best_path)
            best_tag = " (best)"
        else:
            best_tag = ""

        torch.save(ddpm.state_dict(), last_path)

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"LR: {curr_lr:.2e} | "
            f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}"
            f"{best_tag}"
        )

    # ---------------- Curves ----------------
    # Save the training and validation loss trajectories for inspection
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (epsilon prediction)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    curve_path = OUT_DIR / "loss_curve.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()

    # ---------------- Sample grid from val ----------------
    # Save qualitative target-versus-generation comparisons on held-out examples
    grid_path = OUT_DIR / "val_targets_vs_generated.png"
    make_val_grid(ddpm, val_ds, device, grid_path)

    print(f"\nBest model saved to: {best_path}")
    print(f"Last model saved to: {last_path}")
    print(f"Loss curve saved to: {curve_path}")
    print(f"Val target vs generated grid saved to: {grid_path}")


if __name__ == "__main__":
    main()
