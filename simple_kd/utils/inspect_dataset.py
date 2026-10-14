import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def inspect_dataset(
    filepath: str | Path,
    raw_shape: tuple | None = None,
    num_display: int = 10,
):
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
