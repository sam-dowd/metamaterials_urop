#!/usr/bin/env python3
from diffusion_dispersion import ContextUnetFiLM, DDPM
from h5_devectorized import H5WaveguideDataset
from film_cnn import FilmCNN
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
from pathlib import Path
import random
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")


# ---------------- CONFIG ----------------
# Dataset and model checkpoint paths
H5_PATH = "data/repacked_waveguides_devectorized_train_val_test.h5"
CNN_CKPT = "film_cnn_model/best_model.pth"
DIFF_CKPT = "diffusion_dispersion_model/best_model.pth"

# Output directory for evaluation results and visualizations
OUT_DIR = Path("best_diffusion_results")

# Batching parameters
# batch size for diffusion sampling (GPU memory dependent)
DIFF_BATCH = 16
# chunk size for CNN inference over (image, lambda) pairs
CNN_CHUNK = 4096

# Evaluation configuration
MODE_INDEX = 0           # index of mode to evaluate
K = 24                   # number of wavelength samples per waveguide

# Diffusion hyperparameters (must match training configuration)
N_T = 1000
BETAS = (1e-4, 0.02)
BASE = 64
U_DIM = 128
GROUPS = 8
COND_FEAT_DIM = 12

# Transformer configuration (must match training)
D_MODEL = 128
NHEAD = 4
N_LAYERS = 2

# Visualization configuration
N_PLOT = 10              # number of waveguides to visualize
SEED = 42                # random seed for reproducibility

# Generation configuration
N_GEN = 10               # number of generated candidates per waveguide
# ----------------------------------------


def get_waveguide_cmap():
    """
    Constructs a custom colormap for waveguide visualization by cropping
    the 'inferno' colormap to emphasize maroon → orange → yellow tones.
    """
    base = cm.get_cmap("inferno")
    cropped = LinearSegmentedColormap.from_list(
        "inferno_maroon_yellow",
        base(np.linspace(0.2, 0.9, 256))
    )
    return cropped


def pack_cond(params_24x4: torch.Tensor, modes_24x4: torch.Tensor, weights_24x4: torch.Tensor) -> torch.Tensor:
    """
    Concatenates per-wavelength conditioning features into a single tensor.

    Input:
        params, modes, weights : (B,24,4)

    Output:
        conditioning sequence  : (B,24,12)
    """
    return torch.cat([params_24x4, modes_24x4, weights_24x4], dim=-1)


@torch.no_grad()
def predict_curve_cnn(cnn, patterns, lambdas, device, chunk=4096):
    """
    Predicts dispersion curves using the trained CNN.

    Args:
        patterns : (B,1,32,32)
        lambdas  : (B,24)

    Returns:
        predicted curves : (B,24)
    """
    B, _, H, W = patterns.shape
    T = lambdas.shape[1]

    # Repeat each image across wavelength dimension
    x_rep = patterns.unsqueeze(1).expand(B, T, 1, H, W).reshape(B * T, 1, H, W)
    lam_flat = lambdas.reshape(B * T, 1)

    preds = torch.empty((B * T,), device=device, dtype=torch.float32)

    # Chunked inference to avoid GPU memory overflow
    for s in range(0, B * T, chunk):
        e = min(B * T, s + chunk)
        p = cnn(x_rep[s:e].to(device), lam_flat[s:e].to(device)).squeeze(-1)
        preds[s:e] = p

    return preds.reshape(B, T)


def build_test_waveguide_tensors(ds, K=24):
    """
    Converts devectorized dataset into per-waveguide grouped tensors.

    Groups entries using (waveguide_id, wl_index) into structured arrays.

    Returns:
        wg_ids        : (N,)
        patterns      : (N,1,32,32)
        params        : (N,24,4)
        modes         : (N,24,4)
        weights       : (N,24,4)
    """
    ids = []
    for i in range(len(ds)):
        *_, waveguide_id, _wl = ds[i]
        ids.append(int(waveguide_id))

    wg_ids = np.array(sorted(set(ids)), dtype=np.int64)
    N = len(wg_ids)
    id_to_row = {int(w): i for i, w in enumerate(wg_ids)}

    patterns = np.zeros((N, 1, 32, 32), dtype=np.float32)
    params = np.full((N, K, 4), np.nan, dtype=np.float32)
    modes = np.full((N, K, 4), np.nan, dtype=np.float32)
    weights = np.full((N, K, 4), np.nan, dtype=np.float32)

    # Populate tensors using wl_index ordering
    for i in range(len(ds)):
        x0, par, m, w, waveguide_id, wl_index = ds[i]
        r = id_to_row[int(waveguide_id)]
        j = int(wl_index)
        if 0 <= j < K:
            patterns[r] = x0.numpy()
            params[r, j] = par.numpy()
            modes[r, j] = m.numpy()
            weights[r, j] = w.numpy()

    # Check for missing wavelength entries
    missing = np.isnan(params[..., 0]).sum(axis=1)
    if missing.max() != 0:
        bad = np.where(missing != 0)[0]
        print(
            f"[WARN] {len(bad)} waveguides missing wl slots. Example rows: {bad[:10].tolist()}")

    return (
        torch.from_numpy(wg_ids.copy()),
        torch.from_numpy(patterns),
        torch.from_numpy(params),
        torch.from_numpy(modes),
        torch.from_numpy(weights),
    )


@torch.no_grad()
def main():
    # Set deterministic seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "diffusion_eval_summary_and_perwaveguide.csv"
    out_png = OUT_DIR / "examples_10rows_target_vs_generated_and_curves.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wg_cmap = get_waveguide_cmap()

    # ---- Load and group test dataset ----
    test_item_ds = H5WaveguideDataset(
        h5_path=H5_PATH, split="test", return_meta=True)

    wg_ids, patterns, params_24x4, modes_24x4, weights_24x4 = build_test_waveguide_tensors(
        test_item_ds, K=K)

    N = wg_ids.numel()

    # Extract wavelength values and conditioning sequence
    lambdas = params_24x4[:, :, 0].contiguous()
    c_seq_all = pack_cond(params_24x4, modes_24x4, weights_24x4).contiguous()

    # Target dispersion curve from dataset
    target_curve_all = modes_24x4[:, :, MODE_INDEX].contiguous()

    # ---- Load trained CNN ----
    cnn = FilmCNN(cond_dim=1).to(device)
    cnn.load_state_dict(torch.load(CNN_CKPT, map_location=device))
    cnn.eval()

    # ---- Load trained diffusion model ----
    unet = ContextUnetFiLM(
        in_channels=1,
        base=BASE,
        cond_feat_dim=COND_FEAT_DIM,
        K=K,
        u_dim=U_DIM,
        groups=GROUPS,
        d_model=D_MODEL,
        nhead=NHEAD,
        n_layers=N_LAYERS,
    )

    ddpm = DDPM(nn_model=unet, betas=BETAS, n_T=N_T, device=device).to(device)
    ddpm.load_state_dict(torch.load(DIFF_CKPT, map_location=device))
    ddpm.eval()

    # ---- CNN predictions on original patterns (for comparison) ----
    cnn_curve_orig = predict_curve_cnn(
        cnn,
        patterns.float(),
        lambdas.float(),
        device=device,
        chunk=CNN_CHUNK,
    ).detach().cpu()

    # ---- Diffusion sampling and best-of-N selection ----
    gen_patterns = torch.empty_like(patterns, dtype=torch.float32)
    cnn_curve_best = torch.empty((N, K), dtype=torch.float32)

    for s in range(0, N, DIFF_BATCH):
        e = min(N, s + DIFF_BATCH)
        B = e - s

        c_seq = c_seq_all[s:e].float().to(device, non_blocking=True)
        lam_b = lambdas[s:e].float().to(device, non_blocking=True)
        tgt_b = target_curve_all[s:e].float().to(device, non_blocking=True)

        # Repeat conditioning for multiple generated candidates
        c_rep = c_seq.unsqueeze(1).expand(
            B, N_GEN, K, COND_FEAT_DIM).reshape(B * N_GEN, K, COND_FEAT_DIM)

        # Generate samples via diffusion
        x_gen_flat, _ = ddpm.sample(
            n_sample=(B * N_GEN),
            size=(1, 32, 32),
            device=device,
            c_seq=c_rep,
        )

        x_gen_flat = x_gen_flat.detach()

        # Repeat lambda values accordingly
        lam_rep = lam_b.unsqueeze(1).expand(
            B, N_GEN, K).reshape(B * N_GEN, K)

        # Predict curves for generated samples
        cnn_curve_flat = predict_curve_cnn(
            cnn,
            x_gen_flat,
            lam_rep,
            device=device,
            chunk=CNN_CHUNK,
        )

        x_gen = x_gen_flat.reshape(B, N_GEN, 1, 32, 32)
        cnn_curve = cnn_curve_flat.reshape(B, N_GEN, K)

        # Select best sample via MSE against target curve
        mse = ((cnn_curve - tgt_b.unsqueeze(1)) ** 2).mean(dim=2)
        best_j = torch.argmin(mse, dim=1)

        # Gather best images and curves
        idx = best_j.view(B, 1, 1, 1, 1).expand(B, 1, 1, 32, 32)
        best_patterns = torch.gather(x_gen, dim=1, index=idx).squeeze(1)

        idxc = best_j.view(B, 1, 1).expand(B, 1, K)
        best_curves = torch.gather(cnn_curve, dim=1, index=idxc).squeeze(1)

        gen_patterns[s:e] = best_patterns.detach().cpu()
        cnn_curve_best[s:e] = best_curves.detach().cpu()

        if s == 0 or (s // DIFF_BATCH) % 10 == 0:
            print(f"[DIFF] sampled+selected {e}/{N} (N_GEN={N_GEN})")

    # ---- Compute evaluation metrics ----
    per_wg_curve_mse = (
        (cnn_curve_best - target_curve_all.cpu()) ** 2).mean(dim=1).numpy()
    diff = (gen_patterns - patterns.float()).numpy()
    per_wg_pixel_mse = (diff ** 2).mean(axis=(1, 2, 3))

    avg_curve_mse = float(per_wg_curve_mse.mean())
    avg_pixel_mse = float(per_wg_pixel_mse.mean())

    # ---- Save CSV results ----
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow(["metric", "value"])
        w.writerow(["avg_curve_mse_per_waveguide", f"{avg_curve_mse:.12g}"])
        w.writerow(["avg_pixel_mse_per_waveguide", f"{avg_pixel_mse:.12g}"])
        w.writerow(["N_waveguides", int(N)])
        w.writerow(["H5_PATH", H5_PATH])
        w.writerow(["CNN_CKPT", CNN_CKPT])
        w.writerow(["DIFF_CKPT", DIFF_CKPT])
        w.writerow(["MODE_INDEX", MODE_INDEX])
        w.writerow(["DIFF_BATCH", DIFF_BATCH])
        w.writerow(["CNN_CHUNK", CNN_CHUNK])
        w.writerow(["N_GEN", N_GEN])
        w.writerow(
            ["DIFF_ARCH", f"base={BASE},u_dim={U_DIM},groups={GROUPS},d_model={D_MODEL},nhead={NHEAD},n_layers={N_LAYERS},K={K},cond_feat_dim={COND_FEAT_DIM}"])

        w.writerow([])
        w.writerow(["waveguide_id", "curve_mse_cnn(best_gen)_vs_target_curve",
                   "pixel_mse_best_gen_vs_target_img"])

        for i in range(N):
            w.writerow([int(wg_ids[i].item()), float(
                per_wg_curve_mse[i]), float(per_wg_pixel_mse[i])])

    print(f"[SAVE] {out_csv}")

    # ---- Plot qualitative examples ----
    chosen = random.sample(range(N), k=min(N_PLOT, N))
    fig, axes = plt.subplots(len(chosen), 3, figsize=(16, 2.8 * len(chosen)))

    if len(chosen) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, idx0 in enumerate(chosen):
        wid = int(wg_ids[idx0].item())

        tgt_img = patterns[idx0, 0].numpy()
        gen_img = gen_patterns[idx0, 0].numpy()

        x = lambdas[idx0].numpy()
        order = np.argsort(x)

        target_curve = target_curve_all[idx0].numpy()
        curve_from_tgt = cnn_curve_orig[idx0].numpy()
        curve_from_best = cnn_curve_best[idx0].numpy()

        vmin = min(float(tgt_img.min()), float(gen_img.min()))
        vmax = max(float(tgt_img.max()), float(gen_img.max()))

        ax = axes[r, 0]
        ax.imshow(tgt_img, cmap=wg_cmap, origin="upper",
                  interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"wg {wid} | TARGET", fontsize=10)
        ax.set_axis_off()

        ax = axes[r, 1]
        ax.imshow(gen_img, cmap=wg_cmap, origin="upper",
                  interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"BEST of {N_GEN}", fontsize=10)
        ax.set_axis_off()

        ax = axes[r, 2]
        ax.plot(x[order], target_curve[order], linewidth=1.5,
                label="Target curve (dataset)")
        ax.plot(x[order], curve_from_tgt[order],
                linewidth=1.5, label="CNN(target img)")
        ax.plot(x[order], curve_from_best[order],
                linewidth=1.5, label="CNN(best generated img)")
        ax.grid(True, alpha=0.35)
        ax.set_title(
            f"Curves | curve_mse={per_wg_curve_mse[idx0]:.3g}", fontsize=10)
        ax.set_xlabel("lambda", fontsize=9)
        ax.set_ylabel(f"mode[{MODE_INDEX}] / pred", fontsize=9)
        ax.tick_params(labelsize=8)

        if r == 0:
            ax.legend(fontsize=8, loc="best", frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[SAVE] {out_png}")


if __name__ == "__main__":
    main()
