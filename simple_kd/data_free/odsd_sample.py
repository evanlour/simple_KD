import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def plot_sampling_profile(
    p_i: torch.Tensor | np.ndarray,
    o_i: torch.Tensor | np.ndarray,
    d_c: torch.Tensor | np.ndarray,
    sampling_score: torch.Tensor | np.ndarray,
    keep_ratio: float,
    savename: Path | None = None,
    n_bins: int = 100,
) -> None:
    """Plots and saves the sample score profile and components for ODSD.

    Visualizes the contribution of confidence probability (p_i), prototype similarity penalty (-o_i),
    density score (d_c), and net sampling score across binned percentiles.

    Args:
        p_i (torch.Tensor | np.ndarray): Confidence probability scores.
        o_i (torch.Tensor | np.ndarray): Prototype similarity / outlier scores.
        d_c (torch.Tensor | np.ndarray): Class density scores.
        sampling_score (torch.Tensor | np.ndarray): Total computed sampling scores.
        keep_ratio (float): The fraction of data retained by the cutoff.
        savename (Path | str): File path where the plot PNG will be saved.
        n_bins (int, optional): Number of histogram bins. Defaults to 100.
    """
    if isinstance(p_i, torch.Tensor):
        p_i = p_i.detach().cpu().numpy()
    if isinstance(o_i, torch.Tensor):
        o_i = o_i.detach().cpu().numpy()
    if isinstance(d_c, torch.Tensor):
        d_c = d_c.detach().cpu().numpy()
    if isinstance(sampling_score, torch.Tensor):
        sampling_score = sampling_score.detach().cpu().numpy()

    # Sort everything by total sampling score
    sort_idx = np.argsort(sampling_score)
    p_sorted, o_sorted, d_sorted = p_i[sort_idx], o_i[sort_idx], d_c[sort_idx]
    score_sorted = sampling_score[sort_idx]

    b = len(sampling_score) // n_bins

    p_bin = [p_sorted[i * b : (i + 1) * b].mean() for i in range(n_bins)]
    o_bin = [o_sorted[i * b : (i + 1) * b].mean() for i in range(n_bins)]
    d_bin = [d_sorted[i * b : (i + 1) * b].mean() for i in range(n_bins)]
    s_bin = [score_sorted[i * b : (i + 1) * b].mean() for i in range(n_bins)]

    plt.figure(figsize=(12, 6))
    idx = np.arange(n_bins)

    # Stack the positive components and plot the penalty below zero
    plt.bar(idx, p_bin, label="Confidence Probability (p_i)", color="blue")
    plt.bar(idx, d_bin, bottom=p_bin, label="Density Score (d_c)", color="green")
    plt.bar(idx, [-x for x in o_bin], label="Prototype Similarity (-o_i)", color="red")

    # The actual score line
    plt.plot(idx, s_bin, color="gold", label="Net Score", linewidth=2)

    # The Cutoff Line
    cutoff_x = int(n_bins * keep_ratio)
    plt.axvline(x=cutoff_x, color="black", linestyle="--", label="Keep Cutoff")

    plt.title(f"ODSD Sampling Profile (Top {int(keep_ratio * 100)}%)")
    plt.xlabel("Sample Score Percentile Bins")
    plt.ylabel("Score Contribution")
    plt.legend(loc="best")

    if savename is not None:
        savename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savename, dpi=300, bbox_inches="tight")
        plt.close()


def odsd_sample(
    teacher_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    augmentations: torch.nn.Module | None = None,
    keep_ratio: float = 0.5,
    kmeans_size: int = 7,
    reverse_scores: bool = False,
    save_location: Path | None = None,
    save_prefix: str = "",
    save_plots: bool = True,
) -> np.ndarray:
    """Performs Stage 1 (Adaptive Prototype Pooling / Sampling) of ODSD.

    Evaluates a candidate/proxy dataset with a pre-trained teacher, estimates cluster prototypes
    via K-Means, calculates confidence (p_i), prototype outlier penalty (o_i), and class density (d_c),
    filters the top samples according to net sampling score, and saves the resulting image subset.

    The method is proposed in:
        Sampling to Distill: Knowledge Transfer from Open-World Data at https://arxiv.org/pdf/2307.16601v2

    Args:
        teacher_model (torch.nn.Module): Pre-trained frozen teacher model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing unlabeled proxy candidate images.
        num_classes (int): Number of target classes predicted by the teacher.
        keep_ratio (float, optional): Fraction of data to retain (range 0.0 to 1.0). Defaults to 0.5.
        kmeans_size (int, optional): Number of KMeans clusters per class for prototype similarity. Defaults to 7.
        reverse_scores (bool, optional): If True, retains lowest scoring samples instead of highest. Defaults to False.
        device (torch.device | None, optional): Device on which teacher inference runs. Defaults to GPU if available.
        save_location (Path, optional): Directory where the image will be stored. Defaults to None.
        save_prefix (str, optional): Prefix for saved artifact filenames. Defaults to "".
        save_plots (bool, optional): Whether to generate and save the score breakdown histogram. Defaults to True.

    Returns:
        np.ndarray: The sampled subset of images as a NumPy array.
    """

    all_outputs = []
    all_probs = []
    all_preds = []

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(memory_format=torch.channels_last)

            if augmentations is not None:
                inputs = augmentations(inputs)

            if not inputs.is_contiguous(memory_format=torch.channels_last):
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            teacher_output = teacher_model(inputs)

            teacher_probs = F.softmax(teacher_output, dim=1)
            teacher_preds = torch.argmax(teacher_probs, dim=1)

            all_outputs.append(teacher_output.cpu())
            all_probs.append(teacher_probs.cpu())
            all_preds.append(teacher_preds.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    total_samples = all_outputs.size(0)
    p_i = torch.zeros(total_samples, device="cpu")
    o_i = torch.zeros(total_samples, device="cpu")
    d_c = torch.zeros(total_samples, device="cpu")

    for i in range(num_classes):
        class_mask = all_preds == i
        class_prototypes = all_outputs[class_mask]
        class_indices = class_mask.nonzero(as_tuple=True)[0]

        if class_prototypes.size(0) > 0:
            n_clusters = min(kmeans_size, class_prototypes.size(0))
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
            kmeans.fit(class_prototypes.numpy())

            cluster_centers = torch.tensor(
                kmeans.cluster_centers_,
                dtype=class_prototypes.dtype,
                device=class_prototypes.device,
            )

            prototype_similarity = sum(
                F.cosine_similarity(class_prototypes, center.unsqueeze(0))
                for center in cluster_centers
            )

            intra_class_outlier_mean = torch.sum(
                class_prototypes, dim=1
            ) / class_prototypes.size(0)
            density_score = torch.sqrt(intra_class_outlier_mean) / torch.log(
                torch.tensor(class_prototypes.size(0) + float(num_classes))
            )

            p_i[class_indices] = all_probs[class_indices, i]
            o_i[class_indices] = prototype_similarity
            d_c[class_indices] = density_score

    # Normalize scores
    p_max = p_i.max()
    p_i = torch.div(p_i, p_max if p_max > 0 else 1.0)

    o_max = o_i.max()
    o_i = torch.div(o_i, o_max if o_max > 0 else 1.0)

    d_c = torch.nan_to_num(d_c, nan=0.0)
    d_max = d_c.max()
    d_c = torch.div(d_c, d_max if d_max > 0 else 1.0)

    sampling_score = p_i - o_i + d_c

    num_to_keep = max(1, int(total_samples * keep_ratio))
    is_largest = not reverse_scores
    _, top_indices = torch.topk(sampling_score, num_to_keep, largest=is_largest)
    top_indices_np = top_indices.numpy()

    subset_images = np.asarray(dataloader.dataset.images)[top_indices_np]  # type: ignore

    elapsed = time.perf_counter() - start_time
    print(
        f"ODSD sampling complete: kept {num_to_keep}/{total_samples} samples in {elapsed:.2f}s"
    )

    if save_plots and save_location is not None:
        dataset_prefix = (
            f"{save_prefix}_sampled_dataset.npy"
            if save_prefix
            else "sampled_dataset.npy"
        )
        dataset_savepath = save_location / dataset_prefix
        plot_prefix = (
            f"{save_prefix}_sample_scores.png" if save_prefix else "sample_scores.png"
        )
        plot_savepath = save_location / plot_prefix
        dataset_savepath.parent.mkdir(parents=True, exist_ok=True)

        np.save(dataset_savepath, subset_images)
        plot_cutoff_ratio = keep_ratio if reverse_scores else (1.0 - keep_ratio)
        plot_sampling_profile(
            p_i=p_i,
            o_i=o_i,
            d_c=d_c,
            sampling_score=sampling_score,
            keep_ratio=plot_cutoff_ratio,
            savename=plot_savepath,
        )

    return subset_images
