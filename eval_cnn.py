#!/usr/bin/env python3
from pathlib import Path
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from film_cnn import FilmCNN
from h5_devectorized import H5WaveguideDataset


# ---------------- CONFIG ----------------
# Dataset and checkpoint paths
H5_PATH = "data/repacked_waveguides_devectorized_train_val_test.h5"
CKPT = "scaling_no_groupnorm_t1/best_filmcnn_topmode_80pct.pth"

# Output directory for evaluation artifacts
OUT_DIR = Path("best_cnn_results")

# Data loading configuration
BATCH_SIZE = 256
NUM_WORKERS = 4
PIN_MEMORY = True

# Evaluation configuration
MODE_INDEX = 0          # index of mode to evaluate
TOPW_AGG = "mean"       # aggregation method over wavelengths: "mean" or "max"

# Visualization settings
N_CURVES_PLOT = 36      # number of waveguides to visualize
GRID = 6                # grid dimension for curve plots (GRID x GRID)
SEED = 42               # random seed for reproducible selection
T = 24                  # expected number of wavelength samples per waveguide
# ----------------------------------------


@torch.no_grad()
def main():
    # Create output directory and define output file paths
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_scatter = OUT_DIR / "scatter_perwaveguide_totalloss_vs_topweight.png"
    out_grid = OUT_DIR / "curves_6x6_actual_vs_pred.png"
    out_txt = OUT_DIR / "summary.txt"

    # Select device for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset with metadata (waveguide_id, wl_index)
    test_ds = H5WaveguideDataset(
        h5_path=H5_PATH, split="test", return_meta=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Load trained model checkpoint
    model = FilmCNN(cond_dim=1).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    # -------- Pass 1: accumulate per-waveguide statistics --------
    # Dictionaries indexed by waveguide_id
    sum_sqerr = {}   # total squared error per waveguide (sum over wavelengths)
    count = {}       # number of wavelength samples per waveguide
    sum_topw = {}    # sum of selected mode weights (for mean aggregation)
    max_topw = {}    # max of selected mode weights (for max aggregation)

    for x0, par, modes, wts, waveguide_id, wl_index in test_loader:
        # Move inputs to device
        x0 = x0.float().to(device, non_blocking=True)
        par = par.float().to(device, non_blocking=True)
        modes = modes.float().to(device, non_blocking=True)

        # Extract conditioning variable (wavelength)
        lam = par[:, 0].unsqueeze(1)

        # Select target mode and compute prediction
        target = modes[:, MODE_INDEX]
        pred = model(x0, lam).squeeze(-1)

        # Compute squared error per sample
        sq_err = (pred - target) ** 2

        # Move results to CPU for aggregation
        sq_err = sq_err.detach().cpu().numpy()
        topw = wts[:, MODE_INDEX].detach().cpu().numpy()
        wg = waveguide_id.detach().cpu().numpy()

        # Accumulate statistics per waveguide
        for i in range(len(wg)):
            g = int(wg[i])
            se = float(sq_err[i])
            tw = float(topw[i])

            sum_sqerr[g] = sum_sqerr.get(g, 0.0) + se
            count[g] = count.get(g, 0) + 1

            sum_topw[g] = sum_topw.get(g, 0.0) + tw
            max_topw[g] = max(max_topw.get(g, -np.inf), tw)

    # -------- Aggregate metrics per waveguide --------
    wgs = np.array(sorted(sum_sqerr.keys()), dtype=np.int64)

    # Total loss per waveguide (sum over all wavelength samples)
    per_wg_total_loss = np.array(
        [sum_sqerr[int(g)] for g in wgs], dtype=np.float64)

    # Mean loss across waveguides
    avg_loss_per_waveguide = float(per_wg_total_loss.mean())

    # Also compute global per-item average loss
    total_loss = float(per_wg_total_loss.sum())
    total_items = int(sum(count.values()))
    avg_loss_per_item = total_loss / max(1, total_items)

    # Aggregate top weight per waveguide
    if TOPW_AGG == "mean":
        per_wg_topw = np.array(
            [sum_topw[int(g)] / count[int(g)] for g in wgs], dtype=np.float64)
    elif TOPW_AGG == "max":
        per_wg_topw = np.array(
            [max_topw[int(g)] for g in wgs], dtype=np.float64)
    else:
        raise ValueError("TOPW_AGG must be 'mean' or 'max'")

    # Distribution of number of wavelength samples per waveguide
    counts = np.array([count[int(g)] for g in wgs], dtype=np.int64)
    uniq_counts, uniq_freqs = np.unique(counts, return_counts=True)
    count_dist = dict(zip(uniq_counts.tolist(), uniq_freqs.tolist()))

    # -------- Scatter plot: loss vs top weight --------
    plt.figure(figsize=(6, 5))
    plt.scatter(per_wg_topw, per_wg_total_loss, s=10, alpha=0.5)
    plt.xlabel(f"Top Weight (wts[{MODE_INDEX}]) [{TOPW_AGG} over wl_index]")
    plt.ylabel("Per-waveguide Total Loss (sum over wl_index)")
    plt.title("Per-waveguide Total Loss vs Top Weight (scatter)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_scatter, dpi=200)
    plt.close()
    print(f"[SAVE] {out_scatter}")

    # -------- Select waveguides for curve visualization --------
    rng = random.Random(SEED)
    chosen = rng.sample(list(wgs), k=min(N_CURVES_PLOT, len(wgs)))
    chosen = [int(x) for x in chosen]

    # Map waveguide_id to row index in visualization arrays
    id_to_row = {wg_id: i for i, wg_id in enumerate(chosen)}

    # Preallocate arrays for curve data
    lam_curves = np.full((len(chosen), T), np.nan, dtype=np.float64)
    tgt_curves = np.full((len(chosen), T), np.nan, dtype=np.float64)
    pred_curves = np.full((len(chosen), T), np.nan, dtype=np.float64)

    # Recreate loader for second pass (curve reconstruction)
    test_loader2 = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    for x0, par, modes, wts, waveguide_id, wl_index in test_loader2:
        wg_np = waveguide_id.detach().cpu().numpy()
        wl_np = wl_index.detach().cpu().numpy()

        # Select only samples belonging to chosen waveguides
        mask = np.isin(wg_np, chosen)
        if not mask.any():
            continue

        x0 = x0.float().to(device, non_blocking=True)
        par = par.float().to(device, non_blocking=True)
        modes = modes.float().to(device, non_blocking=True)

        lam = par[:, 0].unsqueeze(1)
        target = modes[:, MODE_INDEX]
        pred = model(x0, lam).squeeze(-1)

        lam_cpu = lam.squeeze(1).detach().cpu().numpy()
        tgt_cpu = target.detach().cpu().numpy()
        pred_cpu = pred.detach().cpu().numpy()

        # Populate per-waveguide curves
        idxs = np.where(mask)[0]
        for i in idxs:
            wg_id = int(wg_np[i])
            wl = int(wl_np[i])
            if 0 <= wl < T and wg_id in id_to_row:
                r = id_to_row[wg_id]
                lam_curves[r, wl] = float(lam_cpu[i])
                tgt_curves[r, wl] = float(tgt_cpu[i])
                pred_curves[r, wl] = float(pred_cpu[i])

    # -------- Plot 6x6 grid of curves --------
    fig, axes = plt.subplots(GRID, GRID, figsize=(14, 14))
    axes = axes.ravel()

    for k, wg_id in enumerate(chosen[: GRID * GRID]):
        ax = axes[k]

        lam_k = lam_curves[k]
        tgt_k = tgt_curves[k]
        pred_k = pred_curves[k]

        # Handle missing lambda values by plotting against index
        if np.any(np.isnan(lam_k)):
            x = np.arange(T)
            ax.plot(x, tgt_k, label="target", linewidth=1)
            ax.plot(x, pred_k, label="pred", linewidth=1)
            ax.set_xlabel("wl_index", fontsize=8)
        else:
            order = np.argsort(lam_k)
            ax.plot(lam_k[order], tgt_k[order], label="target", linewidth=1)
            ax.plot(lam_k[order], pred_k[order], label="pred", linewidth=1)
            ax.set_xlabel("lambda (params[0])", fontsize=8)

        ax.set_title(f"wg {wg_id}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_grid, dpi=200)
    plt.close()
    print(f"[SAVE] {out_grid}")

    # -------- Save summary metrics --------
    summary = (
        f"avg_loss_per_waveguide (total dataset loss / #waveguides): {avg_loss_per_waveguide:.12g}\n"
        f"avg_loss_per_item (total dataset loss / #items): {avg_loss_per_item:.12g}\n"
        f"N_waveguides: {len(wgs)}\n"
        f"N_items: {total_items}\n"
        f"items_per_waveguide_distribution: {count_dist}\n"
        f"MODE_INDEX: {MODE_INDEX}\n"
        f"TOPW_AGG: {TOPW_AGG}\n"
        f"H5_PATH: {H5_PATH}\n"
        f"CKPT: {CKPT}\n"
    )

    out_txt.write_text(summary)
    print(f"[SAVE] {out_txt}")
    print(summary.strip())


if __name__ == "__main__":
    main()
