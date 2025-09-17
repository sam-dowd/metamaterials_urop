# Replace your Dataset with this pattern (no normalization, new H5)

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

def quarter(pattern: np.ndarray) -> np.ndarray:
    return pattern[:32, :32]

class WaveguideDatasetTop1(Dataset):
    def __init__(self, h5_path: str, split: str = "train", indices=None, ensure_quarter: bool = True):
        assert split in ("train", "test")
        self.h5_path = h5_path
        self.split = split
        self.ensure_quarter = ensure_quarter
        self._h5 = None  # lazy-open per worker

        # Discover sizes cheaply
        with h5py.File(self.h5_path, "r") as f:
            self.n_train = f["pattern_train"].shape[0]
            self.n_test  = f["pattern_test"].shape[0]

        n = self.n_train if split == "train" else self.n_test
        self.indices = np.arange(n, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def _h5f(self):
        if self._h5 is None:
            # Each worker gets its own handle (thread/process safe)
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.indices)

    def _keys_for_split(self):
        if self.split == "train":
            return ("pattern_train", "weight_train", "neff_train", "params_train")
        else:
            return ("pattern_test", "weight_test", "neff_test", "params_test")

    def __getitem__(self, idx):
        f = self._h5f()
        pat_key, wt_key, neff_key, par_key = self._keys_for_split()
        real_idx = int(self.indices[idx])

        patt = np.asarray(f[pat_key][real_idx])  # [H,W] or [1,H,W]
        if patt.ndim == 3 and patt.shape[0] == 1:
            patt = patt[0]
        if self.ensure_quarter:
            patt = quarter(patt)
        patt_t = torch.tensor(patt, dtype=torch.float32).unsqueeze(0)  # [1,32,32]

        params_t = torch.tensor(f[par_key][real_idx], dtype=torch.float32)   # [P]
        neff     = float(np.asarray(f[neff_key][real_idx]))
        weight   = float(np.asarray(f[wt_key][real_idx]))
        cond_t   = torch.tensor([neff, weight], dtype=torch.float32)         # [2]

        return cond_t, params_t, patt_t

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
