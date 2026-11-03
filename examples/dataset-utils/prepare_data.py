import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import torchvision
from sklearn.model_selection import train_test_split


def compute_mean_std_uint8(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """This function returns the mean and the std of a dataset.

    Args:
        dataset (np.array): The dataset that the mean and std values will be extracted..

    Returns:
        (np.ndarray, np.ndarray): The mean and std value of the given dataset.
    """
    assert images.dtype == np.uint8, "Input images are not uint8."
    num_channels = images.shape[-1]
    total_pixels = images.shape[0] * images.shape[1] * images.shape[2]

    values = np.arange(256)  # [0-255]
    means = []
    stds = []

    for c in range(num_channels):
        channel_data = images[..., c].ravel()  # returns the flattened channel
        counts = np.bincount(channel_data, minlength=256)
        mean_255 = np.sum(counts * values) / total_pixels
        var_255 = np.sum(counts * ((values - mean_255) ** 2)) / total_pixels
        std_255 = np.sqrt(var_255)

        means.append(mean_255 / 255.0)
        stds.append(std_255 / 255.0)

    return (np.array(means), np.array(stds))


def save_relevant_info(
    name: str,
    trainset: np.ndarray,
    valset: np.ndarray,
    testset: np.ndarray,
    classes: list[str],
    val_split: float,
    filename: Path,
):
    data = {}
    train_stats = compute_mean_std_uint8(trainset)
    val_stats = compute_mean_std_uint8(valset)
    test_stats = compute_mean_std_uint8(testset)
    data[f"{name}_train_mean"] = train_stats[0].tolist()
    data[f"{name}_train_std"] = train_stats[1].tolist()
    data[f"{name}_val_mean"] = val_stats[0].tolist()
    data[f"{name}_val_std"] = val_stats[1].tolist()
    data[f"{name}_test_mean"] = test_stats[0].tolist()
    data[f"{name}_test_std"] = test_stats[1].tolist()
    data[f"{name}_val_split"] = val_split
    data[f"{name}_num_classes"] = len(classes)
    data[f"{name}_classes"] = classes
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_split",
        nargs="?",
        const=0.15,
        default=0.15,
        type=float,
        help="The validation split percentage. Value range: [0.0, 1.0]",
    )
    parser.add_argument(
        "--include_CIFAR100",
        action="store_true",
        help="Include CIFAR100 dataset when creating data",
    )
    parser.add_argument(
        "--save_location",
        nargs="?",
        const=".",
        default=".",
        type=str,
        help="The location where the saved data will go",
    )
    args = parser.parse_args()

    if not (0.0 <= args.validation_split <= 1.0):
        print(
            "Invalid validation split argument given, the default value of 0.15 will be used."
        )
        args.validation_split = 0.15

    save_dir = Path(args.save_location or ".")
    save_dir.mkdir(parents=True, exist_ok=True)

    os.makedirs("./res", exist_ok=True)
    if args.include_CIFAR100:
        CIFAR100_save_dir = save_dir / "CIFAR100"
        CIFAR100_save_dir.mkdir(parents=True, exist_ok=True)

        CIFAR100_trainset = torchvision.datasets.CIFAR100(
            root="./res", train=True, download=True, transform=None
        )
        CIFAR100_testset = torchvision.datasets.CIFAR100(
            root="./res", train=False, download=True, transform=None
        )
        CIFAR100_classes = CIFAR100_trainset.classes

        CIFAR100_train_images = CIFAR100_trainset.data
        CIFAR100_train_labels = np.array(CIFAR100_trainset.targets, dtype=np.int64)
        CIFAR100_test_images = CIFAR100_testset.data
        CIFAR100_test_labels = np.array(CIFAR100_testset.targets, dtype=np.int64)
        (
            CIFAR100_train_images,
            CIFAR100_val_images,
            CIFAR100_train_labels,
            CIFAR100_val_labels,
        ) = train_test_split(
            CIFAR100_train_images,
            CIFAR100_train_labels,
            test_size=args.validation_split,
            random_state=42,
            stratify=CIFAR100_train_labels,
        )

        np.save(CIFAR100_save_dir / "CIFAR100_train_images.npy", CIFAR100_train_images)
        np.save(CIFAR100_save_dir / "CIFAR100_train_labels.npy", CIFAR100_train_labels)
        np.save(CIFAR100_save_dir / "CIFAR100_val_images.npy", CIFAR100_val_images)
        np.save(CIFAR100_save_dir / "CIFAR100_val_labels.npy", CIFAR100_val_labels)
        np.save(CIFAR100_save_dir / "CIFAR100_test_images.npy", CIFAR100_test_images)
        np.save(CIFAR100_save_dir / "CIFAR100_test_labels.npy", CIFAR100_test_labels)

        save_relevant_info(
            "CIFAR100",
            cast(np.ndarray, CIFAR100_train_images),
            cast(np.ndarray, CIFAR100_val_images),
            CIFAR100_test_images,
            CIFAR100_classes,
            args.validation_split,
            CIFAR100_save_dir / "CIFAR100_info.json",
        )
        del (
            CIFAR100_train_images,
            CIFAR100_train_labels,
            CIFAR100_val_images,
            CIFAR100_val_labels,
            CIFAR100_test_images,
            CIFAR100_test_labels,
        )
        gc.collect()

        if Path("./res").exists():
            shutil.rmtree(Path("./res"))
