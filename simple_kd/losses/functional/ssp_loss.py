import torch
import torch.nn.functional as F


def ssp_loss(student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
            teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
            stage_percentages: tuple[float, ...] | list[float] = (0.25, 0.25, 0.25, 0.25),
            gamma: float = 4.0,
            sort_channels: bool = True) -> torch.Tensor:
    """Simple feature based loss for knowledge distillation. Number of stages between teacher and student, along with the length of stage percentages must match.

    This is a method i created based on the Fitnets method (Romero et al, 2014), https://arxiv.org/abs/1412.6550.
    The original code can be found at https://github.com/adri-romsor/FitNets.

    Args:
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): List of intermediate activations for the student model.
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): List of intermediate activations for the teacher model.
        stage_percentages (tuple[float], optional): Tuple of floats representing each stage's contribution. Defaults to (0.25, 0.25, 0.25, 0.25).
        gamma (float, optional): Multiplies the final loss value by gamma. Defaults to 4.0.
        sort_channels (bool, optional): Sorts the flattened dimensions of the student and teacher stages. Speeds up the function by a lot and seems to yield better results.

    Returns:
        torch.Tensor: The distillation loss.
    """
    loss = 0
    for teacher_stage, student_stage, stage_percentage in zip(teacher_stages, student_stages, stage_percentages):
        teacher_flat = F.adaptive_avg_pool2d(teacher_stage, (1, 1)).flatten(1)
        student_flat = F.adaptive_avg_pool2d(student_stage, (1, 1)).flatten(1)

        if sort_channels: # Descending seems to slightly speed up the process
            student_flat, _ = torch.sort(student_flat, dim=1, descending=True)
            teacher_flat, _ = torch.sort(teacher_flat, dim=1, descending=True)

        if teacher_flat.shape[1] > student_flat.shape[1]:
            teacher_flat = F.adaptive_avg_pool1d(teacher_flat.unsqueeze(1), student_flat.shape[1]).squeeze(1)

        if teacher_flat.shape[1] < student_flat.shape[1]:
            student_flat = F.adaptive_avg_pool1d(student_flat.unsqueeze(1), teacher_flat.shape[1]).squeeze(1)

        loss += F.mse_loss(student_flat, teacher_flat, reduction="mean") * stage_percentage
    return loss * gamma # type: ignore
