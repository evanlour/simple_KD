import math

import torch
import torch.nn.functional as F


def _get_correlation_matrix(
    feat: torch.Tensor, gamma: float = 1.0, p_order: int = 2
) -> torch.Tensor:
    feat = F.normalize(feat, p=2, dim=-1)
    sim_mat = torch.matmul(feat, feat.mT)

    # Taylor-truncated Poisson-style kernel: exp(-2\gamma) * Σ_p (2\gamma)^p / p! * s^p
    x = 2 * gamma
    coeffs = [x**p / math.factorial(p) for p in range(p_order + 1)]

    # Horner's method: c0 + s*(c1 + s*(c2 + ...))
    corr_mat = torch.full_like(sim_mat, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        corr_mat = corr_mat * sim_mat + c

    return math.exp(-x) * corr_mat


def cc_loss(
    student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    stage_percentages: tuple[float, ...] | list[float] = (0.1, 0.1, 0.2, 0.6),
    gamma: float = 1.0,
    p_order: int = 2,
) -> torch.Tensor:
    """Moves instance and correlation level information from the teacher to the student.

    Method is from the paper Correlation Congruence for Knowledge Distillation (https://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf).

    Args:
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): List of intermediate activations for the student model.
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): List of intermediate activations for the teacher model.
        stage_percentages (tuple[float], optional): Tuple of floats representing each stage's contribution. Defaults to (0.1, 0.1, 0.2, 0.6).
        gamma (float, optional): Multiplies the final loss value by gamma. Defaults to 1.0.
        p_order (int, optional): The order of the Taylor series. Defaults to 2.

    Returns:
        torch.Tensor: The distillation loss.
    """
    loss = 0.0

    for student_stage, teacher_stage, stage_percentage in zip(
        student_stages, teacher_stages, stage_percentages
    ):
        if teacher_stage.shape[2:] != student_stage.shape[2:]:
            student_stage = F.interpolate(
                student_stage, size=teacher_stage.shape[2:], mode="bilinear"
            )

        student_cc = _get_correlation_matrix(
            student_stage, gamma=gamma, p_order=p_order
        )
        teacher_cc = _get_correlation_matrix(
            student_stage, gamma=gamma, p_order=p_order
        )
        loss += F.mse_loss(student_cc, teacher_cc) * stage_percentage

    return loss * gamma  # type: ignore
