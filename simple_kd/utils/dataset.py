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
    def __init__(self, images_path: str, labels_path: str, loading_mode: Literal['r+', 'r', 'w+', 'c'] | None = 'r', augmentations: torch.nn.Module | None = None):
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
    def __init__(self, images: np.ndarray, labels: np.ndarray, augmentations: torch.nn.Module | None = None):
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
