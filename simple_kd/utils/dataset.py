import bisect
from typing import Literal

import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    """Memory-mapped PyTorch Dataset for loading NumPy (.npy) image and label arrays.

    Args:
        images_path (str): The filesystem path of the images .npy file.
        labels_path (str): The filesystem path of the labels .npy file.
        loading_mode (Literal['r+', 'r', 'w+', 'c'] | None, optional): Memory-map mode passed to np.load. Defaults to 'r'.
        augmentations (torch.nn.Module | None, optional): On-fetch PyTorch data augmentations. Defaults to None.
    """

    def __init__(
        self,
        images_path: str,
        labels_path: str,
        loading_mode: Literal["r+", "r", "w+", "c"] | None = "r",
        augmentations: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.images = np.load(images_path, mmap_mode=loading_mode)
        self.labels = np.load(labels_path, mmap_mode=loading_mode)
        self.augmentations = augmentations

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetches a sample and returns the augmented image and label tensor."""
        image = self.images[index]
        if self.augmentations is not None:
            image = self.augmentations(image)
        label = torch.tensor(self.labels[index]).long()
        return image, label


class InMemoryDataset(torch.utils.data.Dataset):
    """In-memory PyTorch Dataset created directly from in-memory NumPy arrays.

    Args:
        images (np.ndarray): In-memory NumPy array of dataset images.
        labels (np.ndarray): In-memory NumPy array of dataset labels.
        augmentations (torch.nn.Module | None, optional): On-fetch PyTorch data augmentations. Defaults to None.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augmentations: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.images = images
        self.labels = labels
        self.augmentations = augmentations

    def __len__(self):
        """Returns the dataset length."""
        return len(self.labels)

    def __getitem__(self, index):
        """Fetch an item and convert it to a pytorch tensor."""
        image = self.images[index]
        label = self.labels[index]

        if self.augmentations is not None:
            image = self.augmentations(image)
        return image, torch.tensor(label).long().squeeze()


class LabelFreeDataset(torch.utils.data.Dataset):
    """Memory-mapped PyTorch Dataset for loading NumPy (.npy) image only.

    Args:
        images_path (str): The filesystem path of the images .npy file.
        loading_mode (Literal['r+', 'r', 'w+', 'c'] | None, optional): Memory-map mode passed to np.load. Defaults to 'r'.
        augmentations (torch.nn.Module | None, optional): On-fetch PyTorch data augmentations. Defaults to None.
    """

    def __init__(
        self,
        images_path: str,
        loading_mode: Literal["r+", "r", "w+", "c"] | None = "r",
        augmentations: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.images = np.load(images_path, mmap_mode=loading_mode)
        self.augmentations = augmentations

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Fetches a sample and returns the augmented image tensor."""
        image = self.images[index]
        if self.augmentations is not None:
            image = self.augmentations(image)
        return image


class MultiPathLabelFreeDataset(torch.utils.data.Dataset):
    """Memory-mapped PyTorch Dataset for loading NumPy (.npy) images from multiple paths.

    Args:
        images_paths (list[str]): List of filesystem paths to the images .npy files.
        loading_mode (Literal['r+', 'r', 'w+', 'c'] | None, optional): Memory-map mode passed to np.load. Defaults to 'r'.
        augmentations (torch.nn.Module | None, optional): On-fetch PyTorch data augmentations. Defaults to None.
    """

    def __init__(
        self,
        images_paths: list[str],
        loading_mode: Literal["r+", "r", "w+", "c"] | None = "r",
        augmentations: torch.nn.Module | None = None,
    ):
        super().__init__()
        if not images_paths:
            raise ValueError("images_paths must contain at least one path")

        self.images_paths = images_paths
        self.loading_mode = loading_mode
        self.augmentations = augmentations

        # Load all arrays with memory mapping and compute cumulative lengths
        self._arrays = []
        self._cumulative_lengths = []
        total = 0
        for path in images_paths:
            arr = np.load(path, mmap_mode=loading_mode)
            self._arrays.append(arr)
            total += len(arr)
            self._cumulative_lengths.append(total)

        self._total_length = total

    def __len__(self) -> int:
        """Returns the total number of samples across all files."""
        return self._total_length

    def __getitem__(self, index: int) -> torch.Tensor:
        """Fetches a sample and returns the augmented image tensor."""
        if index < 0:
            index += self._total_length
        if index < 0 or index >= self._total_length:
            raise IndexError(
                f"Index {index} out of range for dataset of length {self._total_length}"
            )

        # Find which file this index belongs to using binary search
        file_idx = bisect.bisect_right(self._cumulative_lengths, index)
        if file_idx == 0:
            local_idx = index
        else:
            local_idx = index - self._cumulative_lengths[file_idx - 1]

        image = self._arrays[file_idx][local_idx]

        if self.augmentations is not None:
            image = self.augmentations(image)

        return image

    @property
    def images(self) -> np.ndarray:
        """Concatenated view of all images across files (for compatibility with single-path datasets)."""
        return np.concatenate(self._arrays)
