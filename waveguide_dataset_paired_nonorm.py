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
      cond   : [4, 2]  (row-wise [mode_i, weight_i])  -- RAW values, no normalization
      params : [4]      -- RAW values, no normalization
      pattern: [1, 32, 32] (float32)
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",                   # "train" or "test"
        # optional: can provide an .npz with 'indices'/'indices_test'
        stats_path=None,
    ):
        assert split in ("train", "test")
        self.h5_path = h5_path
        self.split = split

        # Optional indices from stats file (if provided)
        self.indices_train = None
        self.indices_test = None
        if stats_path is not None:
            stats = np.load(stats_path)
            self.indices_train = stats["indices"] if "indices" in stats else None
            self.indices_test = stats["indices_test"] if "indices_test" in stats else None

        # Discover dataset sizes (to build default indices if needed)
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
            return ("pattern_train", "weight_train", "neff_train", "params_train")
        else:
            return ("pattern_test", "weight_test", "neff_test", "params_test")

    def __getitem__(self, idx):
        """
        Returns:
          cond:    torch.FloatTensor [4,2] with columns (mode_i, weight_i) -- RAW
          params:  torch.FloatTensor [4]  -- RAW
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

            # No log/normalize: just stack raw mode & weight
            cond = torch.stack([mode, weight], dim=1)  # [4,2]

        return cond, params, pattern
