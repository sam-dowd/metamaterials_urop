# import h5py
# import numpy as np

# # Path to your data
# h5_path = 'train_test_split.h5'
# print('Starting cleaning')
# with h5py.File(h5_path, 'r') as f:
#     weights = f['weight_train'][:]
#     mask = np.sum(weights, axis=1) < 100
#     indices = np.where(mask)[0]

#     weights_filtered = weights[indices]
#     modes_filtered = f['neff_train'][indices]

#     stats = {
#         'meanw': np.mean(weights_filtered, axis=0),
#         'stdw': np.std(weights_filtered, axis=0),
#         'meanm': np.mean(modes_filtered, axis=0),
#         'stdm': np.std(modes_filtered, axis=0),
#         'indices': indices
#     }

# np.savez("waveguide_stats.npz", **stats)
# print("Saved waveguide_stats.npz")
# import h5py
# import numpy as np
# from tqdm import tqdm


# h5_path = 'train_test_split.h5'
# chunk_size = 10000

# mask_list = []
# total_samples = 0


# def online_mean_std(dataset, indices):
#     n = 0
#     mean = None
#     M2 = None

#     for i in tqdm(indices, desc="Computing mean/std"):
#         x = dataset[i]
#         if mean is None:
#             mean = np.zeros_like(x, dtype=np.float64)
#             M2 = np.zeros_like(x, dtype=np.float64)

#         n += 1
#         delta = x - mean
#         mean += delta / n
#         M2 += delta * (x - mean)

#     var = M2 / (n - 1)
#     std = np.sqrt(var)
#     return mean, std


# def batch_mean_std(dataset, indices, batch_size=10000):
#     data = []
#     for i in tqdm(range(0, len(indices), batch_size), desc="Loading data"):
#         batch_inds = indices[i:i+batch_size]
#         data.append(dataset[batch_inds].astype(np.float64))
#     data = np.concatenate(data, axis=0)
#     mean = np.mean(data, axis=0)
#     std = np.std(data, axis=0)
#     return mean, std


# print("Scanning weights for mask condition...")

# with h5py.File(h5_path, 'r') as f:
#     weight_dset = f['weight_train']
#     n_samples = weight_dset.shape[0]

#     for i in range(0, n_samples, chunk_size):
#         chunk = weight_dset[i:i+chunk_size]
#         total_weights = np.sum(chunk, axis=1)
#         mask_chunk = (total_weights >= 90) & (total_weights <= 100)
#         valid_indices = np.where(mask_chunk)[0] + i  # shift to global index
#         mask_list.append(valid_indices)
#         total_samples += len(valid_indices)
#         print(
#             f"Processed {i}/{n_samples} entries, kept {len(valid_indices)}...")

#     indices = np.concatenate(mask_list)
#     print(f"Final filtered sample count: {len(indices)}")

#     # meanw, stdw = online_mean_std(f['weight_train'], indices)
#     # meanm, stdm = online_mean_std(f['neff_train'], indices)
#     meanw, stdw = batch_mean_std(f['weight_train'], indices)
#     meanm, stdm = batch_mean_std(f['neff_train'], indices)

# np.savez("waveguide_stats_above90.npz", meanw=meanw, stdw=stdw,
#          meanm=meanm, stdm=stdm, indices=indices)
# print("Saved waveguide_stats_above90.npz")
import h5py
import numpy as np
from tqdm import tqdm


h5_path = 'train_test_split.h5'
chunk_size = 10000

mask_list = []
total_samples = 0


def online_mean_std(dataset, indices):
    n = 0
    mean = None
    M2 = None

    for i in tqdm(indices, desc="Computing mean/std"):
        x = dataset[i]
        if mean is None:
            mean = np.zeros_like(x, dtype=np.float64)
            M2 = np.zeros_like(x, dtype=np.float64)

        n += 1
        delta = x - mean
        mean += delta / n
        M2 += delta * (x - mean)

    var = M2 / (n - 1)
    std = np.sqrt(var)
    return mean, std


def batch_mean_std(dataset, indices, batch_size=10000):
    data = []
    for i in tqdm(range(0, len(indices), batch_size), desc="Loading data"):
        batch_inds = indices[i:i+batch_size]
        data.append(dataset[batch_inds].astype(np.float64))
    data = np.concatenate(data, axis=0)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std

print("Scanning weights for mask condition...")

with h5py.File(h5_path, 'r') as f:
    weight_dset = f['weight_train']
    n_samples = weight_dset.shape[0]

    for i in range(0, n_samples, chunk_size):
        chunk = weight_dset[i:i+chunk_size]
        mask_chunk = np.sum(chunk, axis=1) < 100
        valid_indices = np.where(mask_chunk)[0] + i  # shift to global index
        mask_list.append(valid_indices)
        total_samples += len(valid_indices)
        print(
            f"Processed {i}/{n_samples} entries, kept {len(valid_indices)}...")

    indices = np.concatenate(mask_list)
    print(f"Final filtered sample count: {len(indices)}")

    # meanw, stdw = online_mean_std(f['weight_train'], indices)
    # meanm, stdm = online_mean_std(f['neff_train'], indices)
    meanw, stdw = batch_mean_std(f['weight_train'], indices)
    meanm, stdm = batch_mean_std(f['neff_train'], indices)
    meanp, stdp = batch_mean_std(f['params_train'], indices)

np.savez("waveguide_stats_params_norm.npz", meanw=meanw, stdw=stdw,
         meanm=meanm, stdm=stdm, meanp=meanp, stdp=stdp, indices=indices)
print("Saved waveguide_stats_params_norm.npz")
