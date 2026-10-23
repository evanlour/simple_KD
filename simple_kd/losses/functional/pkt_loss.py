import torch


def _retrieve_probs(output: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inner helper function used to extract probabilities from the given model output."""
    if output.dim() > 2:
        output = output.flatten(start_dim=1)
    # Normalize each vector by its norm
    features_norm = torch.sqrt(torch.sum(output**2, dim=1, keepdim=True))
    features = output / (features_norm + eps)
    features = features.nan_to_num(nan=0.0)  # Removes NaN values
    features_cos = torch.mm(
        features, features.transpose(0, 1)
    )  # Find cosine similarity
    features_cos = (features_cos + 1.0) / 2.0  # Normalize to [0, 1]
    probs = features_cos / (torch.sum(features_cos, dim=1, keepdim=True) + eps)
    return probs


def _single_pkt_loss(
    student_stage: torch.Tensor, teacher_stage: torch.Tensor, eps: float = 1e-6
):
    """Helper to calculate the probabilities for a single tensor couple."""
    student_probs = _retrieve_probs(student_stage, eps=eps)
    teacher_probs = _retrieve_probs(teacher_stage, eps=eps)
    return torch.mean(
        teacher_probs * torch.log((teacher_probs + eps) / (student_probs + eps))
    )


def pkt_loss(
    student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    stage_percentages: list[float] | tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    gamma: float = 256.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Code implementation of the Probabilistic Knowledge Transfer (PKT) method.

    The original paper Learning Deep Representations with Probabilistic Knowledge Transfer can be found at https://arxiv.org/pdf/1803.10837.

    Code implementation was mostly extracted from the Knowledge Distillation Zoo repository https://github.com/AberHu/Knowledge-Distillation-Zoo
    along with the mdistiller repository at https://github.com/megvii-research/mdistiller.

    Args:
            student_stages list[torch.Tensor] | tuple[torch.Tensor, ...]: The student model output.
            teacher_stages list[torch.Tensor] | tuple[torch.Tensor, ...]: The teacher model output.
            stage_percentages (list[float] | tuple[float, ...], optional): Tuple of floats representing each stage's contribution. Defaults to (0.25, 0.25, 0.25, 0.25).
            gamma (float, optional): Loss multiplier for the method. Defaults to 256.0.
            eps (float, optional): Prevents division by zero. Defaults to 1e-6.

    Returns:
            torch.Tensor: The distillation loss.
    """
    if len(teacher_stages) != len(student_stages):
        raise ValueError(
            f"Sizes of teacher stages ({len(teacher_stages)}) and student stages ({len(student_stages)}) do not match."
        )

    loss = 0
    for student_stage, teacher_stage, stage_percentage in zip(
        student_stages, teacher_stages, stage_percentages
    ):
        loss += (
            _single_pkt_loss(
                student_stage=student_stage, teacher_stage=teacher_stage, eps=eps
            )
            * stage_percentage
        )
    return loss * gamma  # type: ignore
