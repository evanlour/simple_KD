import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def inspect_dataset(
    filepath: str | Path,
    raw_shape: tuple | None = None,
    num_display: int = 10,
):
    """Inspects a dataset file, printing statistics (shape, intensity, duplicate hash count) and displaying sample images.

    Args:
        filepath (str | Path): Path to the dataset file (.npy or raw binary memmap).
        raw_shape (tuple, optional): Required shape (N, H, W, C) if reading a raw binary file without .npy headers. Defaults to None.
        num_display (int, optional): Number of evenly spaced sample images to plot using matplotlib. Defaults to 10.

    Raises:
        ValueError: If loading a raw memmap without raw_shape.
    """
    path = Path(filepath)

    try:
        data = np.load(path, mmap_mode="r")
    except (ValueError, OSError):
        if raw_shape is None:
            raise ValueError(
                "Raw memmap without .npy header requires raw_shape=(N, H, W, C)."
            )
        data = np.memmap(path, dtype="uint8", mode="r", shape=raw_shape)

    total = len(data)

    p_min, p_max, p_mean = data.min(), data.max(), data.mean()

    hashes = [hashlib.sha256(img.tobytes()).hexdigest() for img in data]
    unique_count = len(set(hashes))

    print(f"File:        {path.name}")
    print(f"Shape:       {data.shape} ({data.dtype})")
    print(f"Intensity:   Min={p_min}, Max={p_max}, Mean={p_mean:.2f}")
    print(
        f"Unique:      {unique_count} / {total} (Duplicates: {total - unique_count})"
    )

    if p_min == p_max:
        print("[WARNING] All pixels are identical across the dataset.")

    if num_display > 0 and total > 0:
        k = min(num_display, total)
        idxs = np.linspace(0, total - 1, k, dtype=int)

        fig, axes = plt.subplots(1, k, figsize=(2 * k, 2.5))
        axes = [axes] if k == 1 else axes

        for ax, idx in zip(axes, idxs):
            ax.imshow(data[idx])
            ax.set_title(f"Idx {idx}", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        plt.show()
