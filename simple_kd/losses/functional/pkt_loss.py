import torch


def _retrieve_probs(output: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """Inner helper function used to extract probabilities from the given model output."""
    # Normalize each vector by its norm
    features_norm = torch.sqrt(torch.sum(output ** 2, dim=1, keepdim=True))
    features = output / (features_norm + eps)
    features = features.nan_to_num(nan=0.0)                     # Removes NaN values
    features_cos = torch.mm(features, features.transpose(0, 1)) # Find cosine similarity
    features_cos = (features_cos + 1.0) / 2.0                   # Normalize to [0, 1]
    probs = features_cos / torch.sum(features_cos, dim=1, keepdim=True)
    return probs

def pkt_loss(student_output: torch.Tensor,
            teacher_output: torch.Tensor,
            gamma: float = 256.0,
            eps: float = 1e-6) -> torch.Tensor:
    """Code implementation of the Probabilistic Knowledge Transfer (PKT) method.

    The original paper Learning Deep Representations with Probabilistic Knowledge Transfer can be found at https://arxiv.org/pdf/1803.10837.

    Code implementation was mostly extracted from the Knowledge Distillation Zoo repository https://github.com/AberHu/Knowledge-Distillation-Zoo
    along with the mdistiller repository at https://github.com/megvii-research/mdistiller.

    Args:
            student_output (torch.Tensor): The student model output.
            teacher_output (torch.Tensor): The teacher model output.
            gamma (float, optional): Loss multiplier for the method. Defaults to 256.0.
            eps (float, optional): Prevents division by zero. Defaults to 1e-6.

    Returns:
            torch.Tensor: The distillation loss.
    """

    teacher_prob = _retrieve_probs(teacher_output, eps=eps)
    student_prob = _retrieve_probs(student_output, eps=eps)
    loss = torch.mean(teacher_prob * torch.log((teacher_prob + eps) / (student_prob + eps))) # KL Divergance
    return loss * gamma
