import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


def quarter(pattern: np.ndarray) -> np.ndarray:
    """Crop top-left quadrant to 32x32 for CNN/diffusion input."""
    return pattern[:32, :32]


class WaveguideDatasetPaired(Dataset):
    """
    Paired waveguide dataset (train/test) that yields a full dispersion curve.

    Output per item:
      cond   : [100, 2]   columns = [mode, weight]               (RAW float32)
      params : [100, 4]   columns = [λ, lattice_norm, n_atom, n_substrate] (RAW float32)
      pattern: [1, 32, 32] quartered from stored 64x64 (float32)

    Notes:
      - Uses `mode_pred_grid_{split}`, `weight_pred_grid_{split}`, and `params_grid_{split}`.
      - Stored patterns remain 64x64 in the H5; quartering is applied only to the returned tensor.
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",        # "train" or "test"
        stats_path=None,             # optional .npz with 'indices' / 'indices_test'
    ):
        assert split in ("train", "test")
        self.h5_path = h5_path
        self.split = split

        # Optional subset indices
        self.indices_train = None
        self.indices_test = None
        if stats_path is not None:
            stats = np.load(stats_path)
            self.indices_train = stats["indices"] if "indices" in stats else None
            self.indices_test = stats["indices_test"] if "indices_test" in stats else None

        # Discover sizes
        with h5py.File(self.h5_path, "r") as f:
            self.n_train = f["pattern_train"].shape[0]
            self.n_test = f["pattern_test"].shape[0]

        if self.split == "train":
            self.indices = (
                self.indices_train
                if self.indices_train is not None
                else np.arange(self.n_train, dtype=np.int64)
            )
        else:  # test
            self.indices = (
                self.indices_test
                if self.indices_test is not None
                else np.arange(self.n_test, dtype=np.int64)
            )

    def __len__(self):
        return len(self.indices)

    def _keys_for_split(self):
        if self.split == "train":
            return (
                "pattern_train",
                "weight_pred_grid_train",
                "mode_pred_grid_train",
                "params_grid_train",
            )
        else:
            return (
                "pattern_test",
                "weight_pred_grid_test",
                "mode_pred_grid_test",
                "params_grid_test",
            )

    def __getitem__(self, idx):
        """
        Returns:
          cond   : torch.FloatTensor [100, 2]   (mode, weight)
          params : torch.FloatTensor [100, 4]   (λ, lattice_norm, n_atom, n_substrate)
          pattern: torch.FloatTensor [1, 32, 32]
        """
        with h5py.File(self.h5_path, "r") as f:
            pat_key, wt_grid_key, mode_grid_key, par_grid_key = self._keys_for_split()
            real_idx = int(self.indices[idx])

            # Pattern (stored 64x64 int16) → quarter → [1,32,32] float32
            pattern_64 = f[pat_key][real_idx]               # [64,64] or [H,W]
            pattern = torch.tensor(quarter(pattern_64),
                                   dtype=torch.float32).unsqueeze(0)

            # Dispersion curves
            weight_vec = torch.tensor(
                f[wt_grid_key][real_idx],  dtype=torch.float32)     # [100]
            mode_vec = torch.tensor(
                f[mode_grid_key][real_idx], dtype=torch.float32)     # [100]
            params = torch.tensor(
                f[par_grid_key][real_idx],  dtype=torch.float32)     # [100,4]

            # Stack (mode, weight) per wavelength → [100,2]
            cond = torch.stack([mode_vec, weight_vec], dim=1)

        return cond, params, pattern
