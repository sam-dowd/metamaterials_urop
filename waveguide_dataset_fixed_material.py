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
      cond   : [4, 2]  (rows identical = [mode_pred, weight_pred]) -- RAW values
      params : [4]      -- RAW values, no normalization
      pattern: [1, 32, 32] (float32)
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",    # "train" or "test"
        stats_path=None,         # optional; only used for indices if provided
    ):
        assert split in ("train", "test")
        self.h5_path = h5_path
        self.split = split

        # Optional sampling indices (if you keep using an .npz with indices)
        self.indices_train = None
        self.indices_test  = None
        if stats_path is not None:
            stats = np.load(stats_path)
            self.indices_train = stats["indices"] if "indices" in stats else None
            self.indices_test  = stats["indices_test"] if "indices_test" in stats else None

        with h5py.File(self.h5_path, "r") as f:
            self.n_train = f["pattern_train"].shape[0]
            self.n_test  = f["pattern_test"].shape[0]

        if self.split == "train":
            self.indices = (
                self.indices_train
                if self.indices_train is not None
                else np.arange(self.n_train, dtype=np.int64)
            )
        else:
            self.indices = (
                self.indices_test
                if self.indices_test is not None
                else np.arange(self.n_test, dtype=np.int64)
            )

    def __len__(self):
        return len(self.indices)

    def _keys_for_split(self):
        """Return keys for this split in the NEW file layout."""
        if self.split == "train":
            # Use predicted top-1 values for conditional
            return ("pattern_train", "weight_pred_train", "mode_pred_train", "params_train")
        else:
            return ("pattern_test",  "weight_pred_test",  "mode_pred_test",  "params_test")

    def __getitem__(self, idx):
        """
        Returns:
          cond:    torch.FloatTensor [4,2] with rows all = [mode_pred, weight_pred] (RAW)
          params:  torch.FloatTensor [4]  -- RAW
          pattern: torch.FloatTensor [1,32,32]
        """
        with h5py.File(self.h5_path, "r") as f:
            pat_key, wt_pred_key, mode_pred_key, par_key = self._keys_for_split()
            real_idx = int(self.indices[idx])

            # quarter pattern to 32x32 -> [1,32,32]
            pattern = torch.tensor(
                quarter(f[pat_key][real_idx]), dtype=torch.float32
            ).unsqueeze(0)

            # params raw -> [4]
            params = torch.tensor(f[par_key][real_idx], dtype=torch.float32)

            # predicted top-1 (scalars)
            weight_pred = float(f[wt_pred_key][real_idx])
            mode_pred   = float(f[mode_pred_key][real_idx])

            # Build cond as [4,2] with identical rows = [mode_pred, weight_pred]
            # (Keeps compatibility with code that expects [4,2] and reads row 0.)
            pair = torch.tensor([mode_pred, weight_pred], dtype=torch.float32)
            cond = pair.repeat(4, 1)  # [4,2]

        return cond, params, pattern
