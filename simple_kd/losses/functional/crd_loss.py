import torch


def _contrast_loss(
    x: torch.Tensor,
    n_data: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Contrastive loss.

    Args:
        x (torch.Tensor): Scores of shape (batch, K + 1), already divided by Z. Column 0 holds the positive pair, the rest the K negatives.
        n_data (int): Number of samples in the training set.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-7.

    Returns:
        torch.Tensor: The contrastive loss.
    """
    batch_size = x.size(0)
    n_negatives = x.size(1) - 1
    p_noise = 1.0 / float(n_data)

    # loss for the positive pair
    pos_pair = x.select(1, 0)
    log_pos = torch.log(pos_pair / (pos_pair + n_negatives * p_noise + eps))

    # loss for the K negative pairs
    neg_pair = x.narrow(1, 1, n_negatives)
    log_neg = torch.log(
        torch.full_like(neg_pair, n_negatives * p_noise)
        / (neg_pair + n_negatives * p_noise + eps)
    )

    return -(log_pos.sum() + log_neg.sum()) / batch_size
