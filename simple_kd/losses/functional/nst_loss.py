import torch
import torch.nn.functional as F


def nst_loss(
    student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    stage_percentages: tuple[float, ...] | list[float] = (0.1, 0.1, 0.3, 0.6),
    gamma: float = 256.0,
) -> torch.Tensor:
    """Computes the Neuron Selectivity Transfer (NST) loss.

    The NST loss encourages the student model to have similar neuron selectivity patterns
    to the teacher model by computing a polykernel distance between normalized feature maps.

    The original paper is "Like What You Like: Knowledge Distill via Neuron Selectivity Transfer", https://arxiv.org/pdf/1707.01219.

    Args:
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate feature maps from student model.
            Each tensor should be of shape (batch_size, channels, height, width).
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate feature maps from teacher model.
            Each tensor should be of shape (batch_size, channels, height, width).
        stage_percentages (tuple[float, ...] | list[float], optional): Per-stage weight fractions for NST loss.
            Defaults to (0.1, 0.1, 0.3, 0.6).
        gamma (float, optional): Multiplies the final loss value by gamma. Defaults to 256.0.

    Returns:
        torch.Tensor: The NST loss, which is the polykernel distance between student and teacher feature maps.
    """
    loss = 0.0

    for student_stage, teacher_stage, stage_percentage in zip(
        student_stages, teacher_stages, stage_percentages
    ):
        if teacher_stage.shape[2:] != student_stage.shape[2:]:
            student_stage = F.interpolate(
                student_stage, size=teacher_stage.shape[2:], mode="bilinear"
            )

        student_map = student_stage.view(
            student_stage.size(0), student_stage.size(1), -1
        )
        teacher_map = teacher_stage.view(
            teacher_stage.size(0), teacher_stage.size(1), -1
        )

        student_map = F.normalize(student_map, dim=2)
        teacher_map = F.normalize(teacher_map, dim=2)

        # Compute polykernel distance
        stage_loss = (
            _poly_kernel(teacher_map, teacher_map).mean()
            + _poly_kernel(student_map, student_map).mean()
            - 2 * _poly_kernel(student_map, teacher_map).mean()
        )
        loss += stage_loss * stage_percentage

    return loss * gamma  # type: ignore


def _poly_kernel(feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
    """Computes the polynomial kernel between two feature maps.

    Args:
        feat1 (torch.Tensor): First feature map of shape (batch, channels, -1).
        feat2 (torch.Tensor): Second feature map of shape (batch, channels, -1).

    Returns:
        torch.Tensor: The polynomial kernel output of shape (batch, 1, 1).
    """
    return torch.bmm(feat2, feat1.transpose(1, 2)).pow(2)
