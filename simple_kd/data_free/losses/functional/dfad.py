import torch


def dfad_loss(
    student_output: torch.Tensor, teacher_output: torch.Tensor
) -> torch.Tensor:
    """Code implementation of the loss of the Data Free Adversarial Distillation (DFAD) method.

    The original paper is "Data-Free Adversarial Distillation" (Fang et al., CVPR 2020), https://arxiv.org/abs/1912.11006.
    The original code can be found at https://github.com/VainF/Data-Free-Adversarial-Distillation.

    Args:
        student_output (torch.Tensor): The student model output.
        teacher_output (torch.Tensor): The teacher model output.

    Returns:
        torch.Tensor: The distillation loss.
    """
    loss = torch.nn.functional.l1_loss(student_output, teacher_output, reduction="mean")
    return loss
