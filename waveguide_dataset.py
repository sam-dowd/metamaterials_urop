import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


def quarter(pattern):
    return pattern[:32, :32]  # downsample to 32x32


class WaveguideDataset(Dataset):
    def __init__(self, h5_path, stats_path="waveguide_stats_log_norm_above75.npz"):
        self.h5_path = h5_path

        # Load precomputed normalization stats and indices
        stats = np.load(stats_path)
        self.meanw = stats['meanw']
        self.stdw = stats['stdw']
        self.meanm = stats['meanm']
        self.stdm = stats['stdm']
        self.meanp = stats['meanp']
        self.stdp = stats['stdp']
        self.indices = stats['indices']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):  # cond and params log normalized
        with h5py.File(self.h5_path, 'r') as f:
            real_idx = self.indices[idx]

            pattern = torch.tensor(
                quarter(f['pattern_train'][real_idx]), dtype=torch.float32).unsqueeze(0)
            weight = torch.tensor(
                f['weight_train'][real_idx], dtype=torch.float32)
            mode = torch.tensor(
                f['neff_train'][real_idx], dtype=torch.float32)
            params = torch.tensor(
                f['params_train'][real_idx], dtype=torch.float32)

            # Apply log1p and normalize
            mode_log = torch.log1p(mode)
            weight_log = torch.log1p(weight)
            params_log = torch.log1p(params)

            mode_norm = (mode_log - torch.tensor(self.meanm)
                         ) / torch.tensor(self.stdm)
            weight_norm = (weight_log - torch.tensor(self.meanw)
                           ) / torch.tensor(self.stdw)
            params_norm = (params_log - torch.tensor(self.meanp)
                           ) / torch.tensor(self.stdp)

            cond = torch.cat([mode_norm, weight_norm], dim=0)

        return cond.float(), params_norm.float(), pattern

    def denormalize_cond(self, cond):
        """
        Reverts normalized cond (modes + weights) back to original scale.
        """
        if cond.dim() == 1:
            mode_log = cond[:4] * \
                torch.tensor(self.stdm) + torch.tensor(self.meanm)
            weight_log = cond[4:] * \
                torch.tensor(self.stdw) + torch.tensor(self.meanw)
        else:
            mode_log = cond[:, :4] * torch.tensor(self.stdm).unsqueeze(
                0) + torch.tensor(self.meanm).unsqueeze(0)
            weight_log = cond[:, 4:] * torch.tensor(self.stdw).unsqueeze(
                0) + torch.tensor(self.meanw).unsqueeze(0)

        mode = torch.expm1(mode_log)
        weight = torch.expm1(weight_log)
        return torch.cat([mode, weight], dim=-1)

    def denormalize_params(self, params):
        """
        Reverts normalized params back to original scale.
        """
        if params.dim() == 1:
            params_log = params * \
                torch.tensor(self.stdp) + torch.tensor(self.meanp)
        else:
            params_log = params * \
                torch.tensor(self.stdp).unsqueeze(0) + \
                torch.tensor(self.meanp).unsqueeze(0)

        return torch.expm1(params_log)
