import torch
import torch.nn.functional as F


def kd_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    T: float = 4.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """The original knowledge distillation process.

    The original paper can be found at https://arxiv.org/pdf/1503.02531.

    Args:
        student_output (torch.Tensor): The student model output.
        teacher_output (torch.Tensor): The teacher model output.
        T (float, optional): The distillation temperature. Defaults to 4.0.
        gamma (float, optional): Gamma multiplier of the method. Defaults to 1.0.

    Returns:
        torch.Tensor: The distillation loss.
    """
    log_soft_student = F.log_softmax(student_output / T, dim=1)
    soft_teacher = F.softmax(teacher_output / T, dim=1)
    lkd_loss = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean")
    return lkd_loss * (T**2) * gamma
