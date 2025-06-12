# # import h5py
# # import torch
# # import numpy as np
# # from torch.utils.data import Dataset


# # def quarter(pattern):
# #     return pattern[:32, :32]  # or however you want to downsample

# # class WaveguideDataset(Dataset):
# #     def __init__(self, h5_path):
# #         self.h5_path = h5_path
# #         with h5py.File(h5_path, 'r') as h5_file:
# #             weights = h5_file['weight_train'][:]
# #             self.mask = np.sum(weights, axis=1) < 100
# #             self.indices = np.where(self.mask)[0]

# #     def __len__(self):
# #         return len(self.indices)

# #     def __getitem__(self, idx):
# #         with h5py.File(self.h5_path, 'r') as f:
# #             real_idx = self.indices[idx]
# #             pattern = torch.tensor(quarter(f['pattern_train'][real_idx]), dtype=torch.float32).unsqueeze(0)
# #             weight = torch.tensor(f['weight_train'][real_idx], dtype=torch.float32)
# #             eigenmode = torch.tensor(f['neff_train'][real_idx], dtype=torch.float32)
# #             params = torch.tensor(f['params_train'][real_idx], dtype=torch.float32)
# #             # Try normalizing
# #             cond = torch.cat([eigenmode, weight], dim=0)
# #         return cond, params, pattern

# import h5py
# import torch
# import numpy as np
# from torch.utils.data import Dataset


# def quarter(pattern):
#     return pattern[:32, :32]  # downsample to 32x32


# class WaveguideDataset(Dataset):
#     def __init__(self, h5_path):
#         self.h5_path = h5_path
#         with h5py.File(h5_path, 'r') as h5_file:
#             weights = h5_file['weight_train'][:]
#             self.mask = np.sum(weights, axis=1) < 100
#             self.indices = np.where(self.mask)[0]

#             # Load all relevant targets (params) for normalization
#             modes = h5_file['neff_train'][self.indices]

#             # Compute mean and std for first 4 and last 4 elements separately
#             self.meanw = np.mean(weights, axis=0)
#             self.stdw = np.std(weights, axis=0)
#             self.meanm = np.mean(modes, axis=0)
#             self.stdm = np.std(modes, axis=0)

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         with h5py.File(self.h5_path, 'r') as f:
#             real_idx = self.indices[idx]

#             pattern = torch.tensor(
#                 quarter(f['pattern_train'][real_idx]), dtype=torch.float32).unsqueeze(0)
#             weight = torch.tensor(
#                 f['weight_train'][real_idx], dtype=torch.float32)
#             mode = torch.tensor(f['neff_train'][real_idx], dtype=torch.float32)
#             # cond = torch.cat([eigenmode, weight], dim=0)

#             params = torch.tensor(
#                 f['params_train'][real_idx], dtype=torch.float32)

#             # Normalize params in two groups
#             weight_norm = (weight - torch.tensor(self.meanw)) / \
#                 torch.tensor(self.stdw)
#             mode_norm = (mode - torch.tensor(self.meanm)) / \
#                 torch.tensor(self.stdw)
#             cond = torch.cat([mode_norm, weight_norm], dim=0)

#         return cond, params, pattern

#     def denormalize_params(self, cond):
#         """
#         Input: normalized tensor [8] or [B, 8]
#         Output: unnormalized tensor with same shape
#         """
#         if cond.dim() == 1:
#             mode_unnorm = cond[:4] * \
#                 torch.tensor(self.std1) + torch.tensor(self.mean1)
#             weight_unnrom = cond[4:] * \
#                 torch.tensor(self.std2) + torch.tensor(self.mean2)
#         else:  # batched
#             mode_unnorm = cond[:, :4] * torch.tensor(self.std1).unsqueeze(
#                 0) + torch.tensor(self.mean1).unsqueeze(0)
#             weight_unnrom = cond[:, 4:] * torch.tensor(self.std2).unsqueeze(
#                 0) + torch.tensor(self.mean2).unsqueeze(0)

#         return torch.cat([mode_unnorm, weight_unnrom], dim=-1)

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

def quarter(pattern):
    return pattern[:32, :32]  # downsample to 32x32


class WaveguideDataset(Dataset):
    def __init__(self, h5_path, stats_path="waveguide_stats_params_norm.npz"):
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

    def __getitem__(self, idx):
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

            # Normalize conditional input (mode + weight)
            weight_norm = (weight - torch.tensor(self.meanw, dtype=torch.float32)) / \
                torch.tensor(self.stdw, dtype=torch.float32)
            mode_norm = (mode - torch.tensor(self.meanm, dtype=torch.float32)) / \
                torch.tensor(self.stdm, dtype=torch.float32)
            cond = torch.cat([mode_norm, weight_norm], dim=0)
            params_norm = (params - torch.tensor(self.meanp, dtype=torch.float32)) / \
                torch.tensor(self.stdp, dtype=torch.float32)

        return cond, params_norm, pattern
    # def __getitem__(self, idx):
    #     with h5py.File(self.h5_path, 'r') as f:
    #         real_idx = self.indices[idx]

    #         pattern = torch.tensor(
    #             quarter(f['pattern_train'][real_idx]), dtype=torch.float32).unsqueeze(0)
    #         weight = torch.tensor(
    #             f['weight_train'][real_idx], dtype=torch.float32)
    #         mode = torch.tensor(
    #             f['neff_train'][real_idx], dtype=torch.float32)
    #         params = torch.tensor(
    #             f['params_train'][real_idx], dtype=torch.float32)

    #         # Pair mode and weight and sort by mode descending
    #         mode_weight_pairs = list(zip(mode.tolist(), weight.tolist()))
    #         # descending sort by mode
    #         mode_weight_pairs.sort(key=lambda x: -x[0])

    #         sorted_mode, sorted_weight = zip(*mode_weight_pairs)
    #         mode = torch.tensor(sorted_mode, dtype=torch.float32)
    #         weight = torch.tensor(sorted_weight, dtype=torch.float32)

    #         # Normalize conditional input (mode + weight)
    #         weight_norm = (weight - torch.tensor(self.meanw, dtype=torch.float32)) / \
    #             torch.tensor(self.stdw, dtype=torch.float32)
    #         mode_norm = (mode - torch.tensor(self.meanm, dtype=torch.float32)) / \
    #             torch.tensor(self.stdm, dtype=torch.float32)
    #         cond = torch.cat([mode_norm, weight_norm], dim=0)

    #     return cond, params, pattern

    # def __getitem__(self, idx):
    #     with h5py.File(self.h5_path, 'r') as f:
    #         real_idx = self.indices[idx]

    #         pattern = torch.tensor(
    #             quarter(f['pattern_train'][real_idx]), dtype=torch.float32).unsqueeze(0)
    #         weight = torch.tensor(
    #             f['weight_train'][real_idx], dtype=torch.float32)
    #         mode = torch.tensor(
    #             f['neff_train'][real_idx], dtype=torch.float32)
    #         params = torch.tensor(
    #             f['params_train'][real_idx], dtype=torch.float32)

    #         # Sort by descending weight while preserving (mode, weight) pairings
    #         sorted_indices = torch.argsort(weight, descending=True)
    #         weight_sorted = weight[sorted_indices]
    #         mode_sorted = mode[sorted_indices]

    #         # Normalize after sorting
    #         weight_norm = (weight_sorted - torch.tensor(self.meanw, dtype=torch.float32)) / \
    #             torch.tensor(self.stdw, dtype=torch.float32)
    #         mode_norm = (mode_sorted - torch.tensor(self.meanm, dtype=torch.float32)) / \
    #             torch.tensor(self.stdm, dtype=torch.float32)

    #         cond = torch.cat([mode_norm, weight_norm], dim=0)

    #     return cond, params, pattern

    def denormalize_cond(self, cond):
        """
        Input: normalized cond tensor [8] or [B, 8]
        Output: unnormalized tensor with same shape
        """
        if cond.dim() == 1:
            mode_unnorm = cond[:4] * \
                torch.tensor(self.stdm, dtype=torch.float32) + \
                torch.tensor(self.meanm, dtype=torch.float32)
            weight_unnorm = cond[4:] * \
                torch.tensor(self.stdw, dtype=torch.float32) + \
                torch.tensor(self.meanw, dtype=torch.float32)
        else:
            mode_unnorm = cond[:, :4] * torch.tensor(self.stdm, dtype=torch.float32).unsqueeze(
                0) + torch.tensor(self.meanm, dtype=torch.float32).unsqueeze(0)
            weight_unnorm = cond[:, 4:] * torch.tensor(self.stdw, dtype=torch.float32).unsqueeze(
                0) + torch.tensor(self.meanw, dtype=torch.float32).unsqueeze(0)

        return torch.cat([mode_unnorm, weight_unnorm], dim=-1)
    
    def denormalize_params(self, params):
        if params.dim() == 1:
            params_unnorm = params * \
                torch.tensor(self.stdp, dtype=torch.float32) + \
                torch.tensor(self.meanp, dtype=torch.float32)
        else:
            params_unnorm = params * torch.tensor(self.stdp, dtype=torch.float32).unsqueeze(
                0) + torch.tensor(self.meanp, dtype=torch.float32).unsqueeze(0)
        return params_unnorm
