import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Default path to the devectorized HDF5 dataset used throughout training and evaluation
H5_PATH = "data/repacked_waveguides_devectorized_train_val_test.h5"


class H5WaveguideDataset(Dataset):
    """
    PyTorch Dataset for devectorized waveguide data stored in HDF5 format.

    Each entry corresponds to a single (waveguide, wavelength) pair. The dataset
    is organized by split ("train", "val", "test") and provides both geometric
    input (waveguide pattern) and associated physical quantities.

    Stored tensors per split:
        pattern      : spatial waveguide layout
        params       : physical parameters (including wavelength)
        modes        : modal values at the given wavelength
        weights      : modal weights
        waveguide_id : identifier for grouping rows into full waveguides (optional)
        wl_index     : index of wavelength within each waveguide (optional)

    Returns:
        x0_t    : waveguide pattern tensor, shape (1, 32, 32)
        par_t   : parameter vector for a single wavelength, shape (4,)
        modes_t : modal values for a single wavelength, shape (4,)
        wts_t   : modal weights for a single wavelength, shape (4,)

    If return_meta=True, also returns:
        gid_t   : waveguide identifier
        wl_t    : wavelength index within the waveguide
    """

    def __init__(self, h5_path=H5_PATH, split="train", ensure_channel_dim=True, return_meta=False):
        # Validate split and store configuration
        assert split in {"train", "val", "test"}
        self.h5_path = str(h5_path)
        self.split = split
        self.ensure_channel_dim = ensure_channel_dim
        self.return_meta = return_meta

        # File handle is opened lazily to support multiprocessing DataLoader workers
        self._h5 = None

        # Determine dataset length without keeping file open
        with h5py.File(self.h5_path, "r") as f:
            g = f[self.split]
            self.len = g["pattern"].shape[0]

    def _ensure_open(self):
        # Lazily open the HDF5 file within each worker process
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest")
            g = self._h5[self.split]

            # Cache dataset handles for efficient indexing
            self._pattern = g["pattern"]
            self._params = g["params"]
            self._modes = g["modes"]
            self._weights = g["weights"]

            # Metadata fields may be absent depending on dataset configuration
            self._gid = g.get("waveguide_id", None)
            self._wl = g.get("wl_index", None)

    def __len__(self):
        # Total number of (waveguide, wavelength) samples
        return self.len

    def __getitem__(self, idx):
        # Ensure file is open in the current worker context
        self._ensure_open()

        # Retrieve raw numpy arrays for a single sample
        pat = self._pattern[idx]
        par = self._params[idx]
        modes = self._modes[idx]
        wts = self._weights[idx]

        # Convert pattern to tensor and enforce channel dimension if necessary
        x0_t = torch.from_numpy(np.array(pat)).float()
        if self.ensure_channel_dim and x0_t.ndim == 2:
            x0_t = x0_t.unsqueeze(0)

        # Convert associated physical quantities to tensors
        par_t = torch.from_numpy(np.array(par)).float()
        modes_t = torch.from_numpy(np.array(modes)).float()
        wts_t = torch.from_numpy(np.array(wts)).float()

        # Return core inputs if metadata is not requested
        if not self.return_meta:
            return x0_t, par_t, modes_t, wts_t

        # Optionally return identifiers for reconstructing full waveguide sequences
        gid_t = torch.tensor(int(
            self._gid[idx]), dtype=torch.long) if self._gid is not None else torch.tensor(-1, dtype=torch.long)
        wl_t = torch.tensor(int(
            self._wl[idx]),  dtype=torch.long) if self._wl is not None else torch.tensor(-1, dtype=torch.long)

        return x0_t, par_t, modes_t, wts_t, gid_t, wl_t


def make_loader(
    h5_path=H5_PATH,
    split="train",
    batch_size=256,
    num_workers=4,
    pin_memory=True,
    ensure_channel_dim=True,
    return_meta=False,
    shuffle=None,
):
    """
    Construct a DataLoader for the devectorized waveguide dataset.

    This function standardizes data loading across training, validation,
    and testing, while exposing key performance-related parameters such
    as parallel workers and memory pinning.

    Args:
        h5_path            : path to HDF5 dataset
        split              : dataset split ("train", "val", "test")
        batch_size         : number of samples per batch
        num_workers        : number of parallel workers for data loading
        pin_memory         : enable pinned memory for faster GPU transfer
        ensure_channel_dim : enforce channel dimension on input patterns
        return_meta        : include waveguide identifiers and indices
        shuffle            : override default shuffling behavior

    Returns:
        DataLoader instance for the specified configuration
    """

    # Instantiate dataset with desired configuration
    ds = H5WaveguideDataset(
        h5_path=h5_path,
        split=split,
        ensure_channel_dim=ensure_channel_dim,
        return_meta=return_meta,
    )

    # Default: shuffle only during training
    if shuffle is None:
        shuffle = (split == "train")

    # Construct DataLoader with standard performance settings
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )
