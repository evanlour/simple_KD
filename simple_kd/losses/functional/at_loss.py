import torch
import torch.nn.functional as F


def _attention_map(stage, power):
    """Helper function in order to find the attention map of the stage"""
    am = torch.pow(torch.abs(stage), power)
    am = torch.sum(am, dim=1, keepdim=True)
    norm = torch.norm(am, dim=(-2,-1), keepdim=True)
    am = torch.div(am, norm + 1e-6)
    return am

def at_loss(student_stages: torch.Tensor,
           teacher_stages: torch.Tensor,
           stage_percentages: tuple[float, ...] | list[float] = (0.25, 0.25, 0.25, 0.25),
           power: float = 2.0,
           gamma: float = 64.0) -> torch.Tensor:
    """Code implementation of the Attention Transfer(AT) method.

    The original paper can be found at https://arxiv.org/pdf/1612.03928.

    Code implementation was originally from https://github.com/AberHu/Knowledge-Distillation-Zoo.

    Similar code can be found at https://github.com/megvii-research/mdistiller/tree/master.

    Args:
        student_stages (list[torch.Tensor]): List of intermediate activations for the student model.
        teacher_stages (list[torch.Tensor]): List of intermediate activations for the teacher model.
        stage_percentages (tuple[float] | list[float], optional): Tuple of floats representing each stage's contribution. Defaults to (0.25, 0.25, 0.25, 0.25).
        power (float, optional): The power the stages are elevated at to retrieve the attention map. Defaults to 2.0.
        gamma (float, optional): Multiplies the final loss value by gamma. Defaults to 64.0.

    Returns:
        torch.Tensor: The distillation loss.
    """

    loss = 0
    for teacher_stage, student_stage, stage_percentage in zip(teacher_stages, student_stages, stage_percentages):
        teacher_flat = _attention_map(teacher_stage, power)
        student_flat = _attention_map(student_stage, power)
        if teacher_flat.shape[2:] != student_flat.shape[2:]:
            student_flat = F.interpolate(student_flat, size=teacher_flat.shape[2:], mode='bilinear')
        loss += F.mse_loss(student_flat, teacher_flat, reduction="mean") * stage_percentage
    return loss * gamma # type: ignore
