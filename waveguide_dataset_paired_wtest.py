import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


def quarter(pattern):
    return pattern[:32, :32]  # crop to 32x32


class WaveguideDatasetPaired(Dataset):
    """
    Paired waveguide dataset that can read either the train or the test split
    from a single HDF5 file and apply the same preprocessing pipeline.

    Output per item:
      cond  : [4, 2]  (row-wise [mode_i_norm, weight_i_norm])
      params: [4]     (normalized)
      pattern: [1, 32, 32]  (float32)
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",                   # "train" or "test"
        stats_path: str = "waveguide_stats_log_norm_above90.npz",
    ):
        assert split in ("train", "test")
        self.h5_path = h5_path
        self.split = split

        # Load normalization stats (should be computed on TRAIN set)
        stats = np.load(stats_path)
        self.meanw = stats["meanw"]
        self.stdw = stats["stdw"]
        self.meanm = stats["meanm"]
        self.stdm = stats["stdm"]
        self.meanp = stats["meanp"]
        self.stdp = stats["stdp"]

        # Indices for sampling
        self.indices_train = stats["indices"] if "indices" in stats else None
        self.indices_test = stats["indices_test"] if "indices_test" in stats else None

        # Discover dataset sizes (to build default indices if needed)
        with h5py.File(self.h5_path, "r") as f:
            self.n_train = f["pattern_train"].shape[0]
            self.n_test = f["pattern_test"].shape[0]

        if self.split == "train":
            if self.indices_train is None:
                # fall back to full train range
                self.indices = np.arange(self.n_train, dtype=np.int64)
            else:
                self.indices = self.indices_train
        else:  # test
            if self.indices_test is None:
                # if not provided, use full test range
                self.indices = np.arange(self.n_test, dtype=np.int64)
            else:
                self.indices = self.indices_test

    def __len__(self):
        return len(self.indices)

    def _keys_for_split(self):
        if self.split == "train":
            return ("pattern_train", "weight_train", "neff_train", "params_train")
        else:
            return ("pattern_test", "weight_test", "neff_test", "params_test")

    def __getitem__(self, idx):
        """
        Returns:
          cond:    torch.FloatTensor [4,2] (normalized log1p of (mode, weight))
          params:  torch.FloatTensor [4]   (normalized log1p)
          pattern: torch.FloatTensor [1,32,32]
        """
        with h5py.File(self.h5_path, "r") as f:
            pat_key, wt_key, neff_key, par_key = self._keys_for_split()
            real_idx = int(self.indices[idx])

            pattern = torch.tensor(
                quarter(f[pat_key][real_idx]), dtype=torch.float32
            ).unsqueeze(0)  # [1,32,32]

            weight = torch.tensor(f[wt_key][real_idx],
                                  dtype=torch.float32)   # [4]
            mode = torch.tensor(f[neff_key][real_idx],
                                dtype=torch.float32)  # [4]
            params = torch.tensor(f[par_key][real_idx],
                                  dtype=torch.float32)  # [4]

            # log1p then normalize
            mode_log = torch.log1p(mode)
            weight_log = torch.log1p(weight)
            params_log = torch.log1p(params)

            mode_norm = (mode_log - torch.tensor(self.meanm, dtype=torch.float32)
                         ) / torch.tensor(self.stdm, dtype=torch.float32)
            weight_norm = (weight_log - torch.tensor(self.meanw, dtype=torch.float32)
                           ) / torch.tensor(self.stdw, dtype=torch.float32)
            params_norm = (params_log - torch.tensor(self.meanp, dtype=torch.float32)
                           ) / torch.tensor(self.stdp, dtype=torch.float32)

            # [4,2] with columns = (mode_i_norm, weight_i_norm)
            cond = torch.stack([mode_norm, weight_norm], dim=1)

        return cond, params_norm, pattern

    # ---------- Denormalizers (unchanged) ----------
    def denormalize_cond(self, cond):
        """
        Reverts normalized cond (shape [4,2] or [B,4,2]) back to original scale.
        """
        meanm = torch.tensor(self.meanm, dtype=torch.float32)
        stdm = torch.tensor(self.stdm, dtype=torch.float32)
        meanw = torch.tensor(self.meanw, dtype=torch.float32)
        stdw = torch.tensor(self.stdw, dtype=torch.float32)

        if cond.dim() == 2:
            mode_log = cond[:, 0] * stdm + meanm
            weight_log = cond[:, 1] * stdw + meanw
        else:
            mode_log = cond[:, :, 0] * stdm.unsqueeze(0) + meanm.unsqueeze(0)
            weight_log = cond[:, :, 1] * stdw.unsqueeze(0) + meanw.unsqueeze(0)

        mode = torch.expm1(mode_log)
        weight = torch.expm1(weight_log)
        return torch.stack([mode, weight], dim=-1)

    def denormalize_params(self, params):
        """
        Reverts normalized params (shape [4] or [B,4]) back to original scale.
        """
        meanp = torch.tensor(self.meanp, dtype=torch.float32)
        stdp = torch.tensor(self.stdp, dtype=torch.float32)

        if params.dim() == 1:
            params_log = params * stdp + meanp
        else:
            params_log = params * stdp.unsqueeze(0) + meanp.unsqueeze(0)

        return torch.expm1(params_log)
