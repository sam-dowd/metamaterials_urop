import h5py
import torch
from torch.utils.data import Dataset, DataLoader

# Default path to the sorted HDF5 dataset containing full waveguide samples
H5_PATH = "data/repacked_waveguides_sorted.h5"


class H5WaveguideDataset(Dataset):
    """
    PyTorch Dataset for the "sorted" waveguide dataset format.

    Each sample corresponds to a full waveguide instance, including:
        - a single spatial pattern (shared across wavelengths)
        - a sequence of wavelength-dependent parameters, modes, and weights

    Dataset structure:
        pattern_{split} : (G, 32, 32)
        params_{split}  : (G, 24, 4)
        modes_{split}   : (G, 24, 4)
        weights_{split} : (G, 24, 4)

    where G is the number of waveguides.

    Returns:
        x0_t    : waveguide pattern, shape (1, 32, 32)
        par_t   : parameter sequence, shape (24, 4)
        modes_t : modal values, shape (24, 4)
        wts_t   : modal weights, shape (24, 4)
    """

    def __init__(self, h5_path=H5_PATH, split="train"):
        # Restrict dataset to supported splits
        assert split in {"train", "test"}

        self.h5_path = h5_path
        self.split = split

        # File handle is opened lazily to support multi-worker DataLoader usage
        self._h5 = None

        # Determine dataset length without keeping the file open
        with h5py.File(self.h5_path, "r") as f:
            self.len = f[f"pattern_{split}"].shape[0]

    def _ensure_open(self):
        # Lazily open HDF5 file inside each worker process
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest")
            s = self.split

            # Cache dataset references for efficient indexing
            self._pattern = self._h5[f"pattern_{s}"]
            self._params = self._h5[f"params_{s}"]
            self._modes = self._h5[f"modes_{s}"]
            self._weights = self._h5[f"weights_{s}"]

    def __len__(self):
        # Total number of waveguide samples
        return self.len

    def __getitem__(self, idx):
        # Ensure file handle is initialized in current worker
        self._ensure_open()

        # Retrieve raw arrays for a single waveguide
        pat = self._pattern[idx]
        par = self._params[idx]
        modes = self._modes[idx]
        wts = self._weights[idx]

        # Convert to torch tensors and enforce channel dimension for CNN input
        x0_t = torch.from_numpy(pat).float().unsqueeze(0)
        par_t = torch.from_numpy(par).float()
        modes_t = torch.from_numpy(modes).float()
        wts_t = torch.from_numpy(wts).float()

        return x0_t, par_t, modes_t, wts_t


def make_loader(h5_path=H5_PATH, split="train", batch_size=256, num_workers=4, pin_memory=True):
    """
    Construct a DataLoader for the sorted waveguide dataset.

    This function standardizes loading behavior across splits and exposes
    key performance parameters such as batch size, parallel workers, and
    memory pinning.

    Args:
        h5_path     : path to HDF5 dataset
        split       : dataset split ("train" or "test")
        batch_size  : number of samples per batch
        num_workers : number of parallel data loading workers
        pin_memory  : enable pinned memory for faster GPU transfer

    Returns:
        Configured DataLoader instance
    """
    ds = H5WaveguideDataset(h5_path, split=split)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False
    )
