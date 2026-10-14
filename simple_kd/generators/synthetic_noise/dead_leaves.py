"""Original code is from https://github.com/mbaradad/learning_with_noise/blob/main/generate_datasets/dead_leaves/generate_dataset.py
Credits to the author there!
"""

import numpy as np
import multiprocessing as mp
from pathlib import Path
import cv2

def dead_leaves(res: int,
    channels: int,
    sigma: int = 3,
    shape_mode: str='mixed',
    max_iters: int=5000,
    rmin: float = 0.03) -> np.ndarray:

    img = np.zeros((res, res, channels), dtype=np.uint8)
    mask = np.zeros((res, res), dtype=np.uint8)  # Single-channel coverage tracker
    total_pixels = res * res

    # Precompute power-law constants for O(1) inverse-CDF sampling
    alpha = 1.0 - sigma
    r_min_pow = rmin ** alpha
    r_pow_range = 1 - r_min_pow
    inv_alpha = 1.0 / alpha

    shapes = ['circle', 'square', 'oriented_square', 'rectangle', 'triangle', 'quadrilater']
    sqrt2_over_2 = np.sqrt(2) / 2

    for i in range(max_iters):
        shape = np.random.choice(shapes) if shape_mode == 'mixed' else shape_mode
        radius = max(int(((r_min_pow + np.random.rand() * r_pow_range) ** inv_alpha) * res), 1)
        color = int(np.random.randint(0, 255)) if channels == 1 else tuple(np.random.randint(0, 255) for _ in range(channels))
        cx, cy = np.array(np.random.uniform(0, res, size=2),dtype='int32')

        if shape == 'circle':
            cv2.circle(img, (cx, cy), radius=radius, color=color, thickness=-1) # type: ignore
            cv2.circle(mask, (cx, cy), radius, 255, thickness=-1) # type: ignore
        else:
            if shape in ('square', 'oriented_square'):
                h = radius * sqrt2_over_2
                corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
                if shape == 'oriented_square':
                    theta = np.random.uniform(0, 2 * np.pi)
                    c, s = np.cos(theta), np.sin(theta)
                    corners = corners @ np.array([[c, s], [-s, c]])
            elif shape == 'rectangle':
                a = np.random.uniform(0, 0.5 * np.pi)
                rx, ry = radius * np.cos(a), radius * np.sin(a)
                corners = np.array([[rx, ry], [rx, -ry], [-rx, -ry], [-rx, ry]])
            else:  # triangle or quadrilater
                n_pts = 3 if shape == 'triangle' else 4
                angles = np.sort(np.random.uniform(0, 2 * np.pi, n_pts))
                corners = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)

            poly = (corners + [cx, cy]).astype(np.int32)[None]
            cv2.fillPoly(img, poly, color) # type: ignore
            cv2.fillPoly(mask, poly, 255) # type: ignore

        if i % 10 == 0 and cv2.countNonZero(mask) == total_pixels:
            break

    img = img.transpose((2,0,1))
    return np.clip(img, 0, 255)

def worker(args):
    idx, resolution, channels, sigma = args
    np.random.seed(idx)
    img = dead_leaves(res=resolution, sigma=sigma, channels=channels)
    return idx, img

def generate_dead_leaves_dataset(res: int, channels: int, savepath: Path, save_prefix: str = "", parallel_workers: int=3, samples: int=50000, sigma: int=3):

    dataset_name = f"{save_prefix}_dead_leaves.npy" if save_prefix else "dead_leaves.npy"
    dataset_savepath = savepath / dataset_name
    dataset_savepath.parent.mkdir(parents=True, exist_ok=True)

    dataset = np.zeros((samples, res, res, channels), dtype=np.uint8)

    args_list = [(i, res, channels, sigma) for i in range(samples)]
    with mp.Pool(processes=parallel_workers) as pool:
        # imap_unordered runs the workers concurrently and yields results as they finish
        for idx, img in pool.imap_unordered(worker, args_list):
            np_imgs = img.transpose(1, 2, 0) # (B, 3, H, W) -> (B, H, W, 3) (channel re-order) # type: ignore
            dataset[idx] = np_imgs
            if (idx + 1) % 1000 == 0:
                print(f"{(idx + 1) * 100 / samples:.1f}% Complete")
    np.save(dataset_savepath, dataset)
