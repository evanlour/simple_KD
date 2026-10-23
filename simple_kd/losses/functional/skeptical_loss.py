import torch
import torch.nn.functional as F


def skeptical_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    aux_classifier: torch.nn.Module,
    T: float = 4.0,
    kd_gamma: float = 0.5,
    in_kd_gamma: float = 1.0,
) -> torch.Tensor:
    """The skeptical loss function used for training from nasty students.

    The original paper can be found here https://openreview.net/pdf?id=Laz0L5tjml.

    The original code implementation by the authors can be found here https://github.com/ksouvik52/Skeptical2021.

    Args:
        student_output (torch.Tensor): The student model output.
        teacher_output (torch.Tensor): The teacher model output.
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): List of intermediate activations for the student model. Only the first will be used, so a list of len equal to 1 can be provided.
        aux_classifier (torch.nn.Module): An auxiliar classifier to match the intermediate model activations. The provided SkepticalClassifier inside the same file can be used.
        T (float, optional): The distillation temperature. Defaults to 4.0.
        kd_gamma (float, optional): The traditional KD part gamma. Defaults to 0.5.
        in_kd_gamma (float, optional): The esoterical KD inside the student model gamma. Defaults to 1.0.

    Returns:
        torch.Tensor: The distillation loss.
    """
    teacher_output = teacher_output.detach()

    log_soft_student = F.log_softmax(student_output / T, dim=1)
    soft_teacher = F.softmax(teacher_output / T, dim=1)
    kd_loss = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean")
    kd_loss = kd_loss * (T**2) * kd_gamma

    pred_aux_student = aux_classifier(student_stages[0].detach())
    pred_aux_student = F.softmax(pred_aux_student / T, dim=1)
    in_lkd_loss = F.kl_div(log_soft_student, pred_aux_student, reduction="batchmean")
    in_lkd_loss = in_lkd_loss * (T**2) * in_kd_gamma
    aux_classifier.zero_grad(set_to_none=True)

    return in_lkd_loss + kd_loss
