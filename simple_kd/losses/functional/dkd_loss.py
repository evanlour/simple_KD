import torch
import torch.nn.functional as F


def _cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def dkd_loss(student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        labels: torch.Tensor,
        T: float = 4.0,
        tckd_gamma: float = 1.0,
        nckd_gamma: float = 8.0) -> torch.Tensor:
    """Code implementation of the Decoupled Knowledge Distillation (DKD) method.

    The original paper Decoupled Knowledge Distillation can be found at https://arxiv.org/pdf/2203.08679.

    Code implementation is mostly extracted from The mdistiller repository at https://github.com/megvii-research/mdistiller.

    Args:
        student_output (torch.Tensor): The student model output.
        teacher_output (torch.Tensor): The teacher model output.
        labels (torch.Tensor): The correct labels for the current batch.
        T (float, optional): The distillation temperature. Defaults to 4.0.
        tckd_gamma (float, optional): The TCKD gamma AKA the matching label knowledge distillation. Defaults to 1.0.
        nckd_gamma (float, optional): The NCKD gamma AKA the non matching label knowledge distillation. Defaults to 8.0.

    Returns:
        torch.Tensor: The distillation loss.
    """

    gt_mask = torch.zeros_like(student_output).scatter_(1, labels.reshape(-1, 1), 1).bool()
    ngt_mask = torch.ones_like(student_output).scatter_(1, labels.reshape(-1, 1), 0).bool()

    soft_teacher = F.softmax(teacher_output / T, dim=1)
    soft_student = F.softmax(student_output / T, dim=1)
    soft_teacher = _cat_mask(soft_teacher, gt_mask, ngt_mask)
    soft_student = _cat_mask(soft_student, gt_mask, ngt_mask)
    log_soft_student = torch.log(soft_student.clamp(min=1e-6))

    nckd_teacher = F.softmax((teacher_output / T) - 1000 * gt_mask, dim=1)
    nckd_student = F.log_softmax((student_output / T) - 1000 * gt_mask, dim=1)

    tckd_loss = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    nckd_loss = F.kl_div(nckd_student, nckd_teacher, reduction="batchmean") * (T ** 2)

    return tckd_gamma * tckd_loss + nckd_gamma * nckd_loss
