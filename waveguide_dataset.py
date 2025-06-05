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
            weight_sums = np.sum(weights, axis=1)
            mask = weight_sums < 100

            self.eigenmodes = torch.tensor(
                h5_file['neff_train'][:][mask], dtype=torch.float32)
            self.weights = torch.tensor(weights[mask], dtype=torch.float32)
            self.params = torch.tensor(
                h5_file['params_train'][:][mask], dtype=torch.float32)
            # Load patterns while file is open
            patterns = h5_file['pattern_train'][:][mask]

        # Now apply the processing after HDF5 file is closed
        self.patterns = torch.tensor(
            np.array([quarter(p) for p in patterns]), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        cond = torch.cat(
            [self.eigenmodes[idx], self.weights[idx]], dim=-1)  # shape (8,)
        params = self.params[idx]
        waveguide = self.patterns[idx]  # shape (1, 32, 32)
        return cond, params, waveguide
