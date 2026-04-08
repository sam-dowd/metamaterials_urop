from film_cnn import FilmCNN
# FiLM-conditioned CNN used to regress dispersion values from waveguide patterns

from h5_devectorized import H5WaveguideDataset
# Dataset providing devectorized waveguide samples with associated physical parameters

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
# Use a non-interactive backend for saving figures in headless environments


# ---------------- Configuration ----------------
H5_PATH = "data/train_test_ordered_devectorized_dataset_structure.csv"
EPOCHS = 240
BATCH_SIZE = 256
LR = 3e-4
NUM_WORKERS = 4
PIN_MEMORY = True
SEED = 42
OUT_DIR = Path("film_cnn_model")
# ----------------------------------------------

K = 24
# Number of wavelength samples per waveguide


def set_seed(seed=42):
    # Configure deterministic behavior for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ItemToLambdaTarget(Dataset):
    """
    Adapter dataset mapping each waveguide/wavelength pair to a supervised regression task.

    Input:
        x0  : waveguide pattern
        lam : wavelength scalar
    Target:
        selected modal value at the given wavelength
    """

    def __init__(self, base_ds, mode_index=0):
        self.base_ds = base_ds
        self.mode_index = mode_index

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x0, par, modes, wts = self.base_ds[idx]
        lam = par[0].unsqueeze(0)
        target = modes[self.mode_index]
        return x0, lam, target


@torch.no_grad()
def eval_epoch_loss(model, loader, device, criterion):
    # Compute mean regression loss over a full dataset split
    model.eval()
    running, count = 0.0, 0

    for x0, lam, target in loader:
        x0 = x0.float().to(device, non_blocking=True)
        lam = lam.float().to(device, non_blocking=True)
        target = target.float().to(device, non_blocking=True)

        pred = model(x0, lam).squeeze(-1)
        loss = criterion(pred, target)

        bs = x0.size(0)
        running += loss.item() * bs
        count += bs

    return running / max(1, count)


@torch.no_grad()
def save_test_curve_grid(model, test_ds_with_meta, device, out_path, n=36, grid=6):
    """
    Visualize predicted vs ground-truth dispersion curves for randomly selected test waveguides.

    For each waveguide, the model is evaluated independently at each wavelength,
    producing a full dispersion curve from scalar-conditioned predictions.
    """
    model.eval()

    gid_to_rows = {}
    for i in range(len(test_ds_with_meta)):
        x0, par, modes, wts, gid, widx = test_ds_with_meta[i]
        gid = int(gid)
        widx = int(widx)
        lam = float(par[0].item())
        tgt = float(modes[0].item())

        if gid not in gid_to_rows:
            gid_to_rows[gid] = {"x0": x0, "lam": {}, "tgt": {}}

        gid_to_rows[gid]["lam"][widx] = lam
        gid_to_rows[gid]["tgt"][widx] = tgt

    gids = sorted(gid_to_rows.keys())
    if len(gids) == 0:
        print("[WARN] No waveguides found in test set for curve grid.")
        return

    g = torch.Generator().manual_seed(SEED + 12345)
    perm = torch.randperm(len(gids), generator=g)[: min(n, len(gids))].tolist()
    chosen_gids = [gids[p] for p in perm]

    fig, axes = plt.subplots(grid, grid, figsize=(14, 14))
    axes = axes.reshape(grid, grid)

    for i, gid in enumerate(chosen_gids):
        row = gid_to_rows[gid]
        x0 = row["x0"].unsqueeze(0).float().to(device)

        lam_np = np.array([row["lam"][k] for k in range(K)], dtype=np.float32)
        tgt_np = np.array([row["tgt"][k] for k in range(K)], dtype=np.float32)

        preds = []
        for k in range(K):
            lam_k = torch.tensor(
                [[lam_np[k]]], dtype=torch.float32, device=device)
            y_k = model(x0, lam_k).squeeze(-1).item()
            preds.append(y_k)

        pred_np = np.array(preds, dtype=np.float32)

        r, c = divmod(i, grid)
        ax = axes[r, c]

        ax.plot(lam_np, tgt_np, "o-", lw=1, ms=2, label="target")
        ax.plot(lam_np, pred_np, "o-", lw=1, ms=2, label="prediction")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"test gid {gid}", fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] {out_path}")


def write_test_mse_csv(out_csv_path: Path, label: str, test_mse: float):
    # Append evaluation results to a CSV file for experiment tracking
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv_path.exists()

    with open(out_csv_path, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["label", "test_mse"])
        if write_header:
            w.writeheader()
        w.writerow({"label": label, "test_mse": f"{test_mse:.8f}"})

    print(f"[SAVE] {out_csv_path} (appended)")


def main():
    # Initialize reproducibility, device, and output directory
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset splits (train/val/test) from preprocessed HDF5 source
    train_base = H5WaveguideDataset(
        h5_path=H5_PATH, split="train", return_meta=False)
    val_base = H5WaveguideDataset(
        h5_path=H5_PATH, split="val", return_meta=False)
    test_base_with_meta = H5WaveguideDataset(
        h5_path=H5_PATH, split="test", return_meta=True
    )

    train_ds = ItemToLambdaTarget(train_base, mode_index=0)
    val_ds = ItemToLambdaTarget(val_base, mode_index=0)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    train_eval_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
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

    test_ds = ItemToLambdaTarget(
        H5WaveguideDataset(h5_path=H5_PATH, split="test", return_meta=False),
        mode_index=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    model = FilmCNN(cond_dim=1, p_drop=0.1).to(device)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_path = OUT_DIR / "best_model.pth"

    # Train model to regress modal values conditioned on wavelength
    for epoch in range(1, EPOCHS + 1):
        model.train()

        for x0, lam, target in train_loader:
            x0 = x0.float().to(device, non_blocking=True)
            lam = lam.float().to(device, non_blocking=True)
            target = target.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x0, lam).squeeze(-1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

        train_loss = eval_epoch_loss(
            model, train_eval_loader, device, criterion)
        val_loss = eval_epoch_loss(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"LR {curr_lr:.2e} | "
            f"Train MSE {train_loss:.6f} | Val MSE {val_loss:.6f} "
            f"{'(best)' if is_best else ''}"
        )

    # Save training and validation loss curves
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curve.png", dpi=150)
    plt.close()

    # Evaluate best checkpoint on test set
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_mse = eval_epoch_loss(model, test_loader, device, criterion)

    print(f"\n[Test] Average MSE: {test_mse:.8f}")

    # Persist evaluation results
    results_csv = OUT_DIR / "test_metrics.csv"
    label = f"FilmCNN_cond1_mode0_seed{SEED}"
    write_test_mse_csv(results_csv, label=label, test_mse=test_mse)

    # Generate qualitative dispersion curve comparisons
    grid_path = OUT_DIR / "test_dispersion_curves_6x6.png"
    save_test_curve_grid(model, test_base_with_meta, device, grid_path)

    print(f"\nBest model saved to: {best_path}")
    print(f"Loss curve saved to: {OUT_DIR / 'loss_curve.png'}")
    print(f"Test curve grid saved to: {grid_path}")
    print(f"Test metrics CSV saved to: {results_csv}")


if __name__ == "__main__":
    main()
