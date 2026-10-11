from typing import Literal

import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    """The initialization of the Dataset class. Looks and loads the relevant image and label npy files. Labels are loaded as longs.

    Args:
        images_path (str): The path of the images file(.npy).
        labels_path (str): The path of the labels file(.npy).
        loading_mode (str, optional): Choose loading mode. Defaults to 'r'.
        augmentations (v2.Compose, optional): Apply pytorch augmentations. Defaults to None.
    """
    def __init__(self, images_path: str, labels_path: str, loading_mode: Literal['r+', 'r', 'w+', 'c'] | None = 'r', augmentations: torch.nn.Module | None = None):
        super().__init__()
        self.images = np.load(images_path, mmap_mode=loading_mode)
        self.labels = np.load(labels_path, mmap_mode=loading_mode)
        self.augmentations = augmentations
    def __len__(self):
        """
        Returns:
            int: The length of self.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """Fetch an item and convert it to a pytorch tensor.
        """
        image = self.images[index]
        if self.augmentations is not None:
            image = self.augmentations(image)
        label = torch.tensor(self.labels[index]).long()
        return image, label

class InMemoryDataset(torch.utils.data.Dataset):
    """The initialization of the Dataset class. Creates the dataset directly from numpy arrays. Labels are loaded as longs.

    Args:
        images_path (str): The path of the images file(.npy).
        labels_path (str): The path of the labels file(.npy).
        loading_mode (str, optional): Choose loading mode. Defaults to 'r'.
        augmentations (v2.Compose, optional): Apply pytorch augmentations. Defaults to None.

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
