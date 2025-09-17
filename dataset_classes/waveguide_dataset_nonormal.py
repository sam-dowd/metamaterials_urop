import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


def quarter(pattern):
    return pattern[:32, :32]  # or however you want to downsample


class WaveguideDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as h5_file:
            weights = h5_file['weight_train'][:]
            self.mask = np.sum(weights, axis=1) < 100
            self.indices = np.where(self.mask)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            real_idx = self.indices[idx]
            pattern = torch.tensor(
                quarter(f['pattern_train'][real_idx]), dtype=torch.float32).unsqueeze(0)
            weight = torch.tensor(
                f['weight_train'][real_idx], dtype=torch.float32)
            eigenmode = torch.tensor(
                f['neff_train'][real_idx], dtype=torch.float32)
            params = torch.tensor(
                f['params_train'][real_idx], dtype=torch.float32)
            # Try normalizing
            cond = torch.cat([eigenmode, weight], dim=0)
        return cond, params, pattern
