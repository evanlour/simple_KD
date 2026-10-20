import torch

from simple_kd.losses.functional import at_loss, kd_loss


def zskt_loss(student_output: torch.Tensor, teacher_output: torch.Tensor,
              student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...], teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
              T: float=4.0, kd_gamma: float=1.0, at_gamma: float=128.0,
              stage_percentages: list[float] | tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)) -> torch.Tensor:
    """Computes the Zero-Shot Knowledge Transfer (ZSKT) loss.

    Combines output knowledge distillation (KL divergence with temperature T)
    and intermediate layer Attention Transfer (AT).

    The original paper is "Zero-shot Knowledge Transfer via Adversarial Belief Matching"
    (Micaelli & Storkey, NeurIPS 2019, https://arxiv.org/abs/1905.09768). The code is available
    here https://github.com/polo5/ZeroShotKnowledgeTransfer


    Args:
        student_output (torch.Tensor): Logits from the student model.
        teacher_output (torch.Tensor): Logits from the teacher model.
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from student model.
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from teacher model.
        T (float, optional): Distillation temperature for KD loss. Defaults to 4.0.
        kd_gamma (float, optional): Multiplier for the KD loss. Defaults to 1.0.
        at_gamma (float, optional): Multiplier for the Attention Transfer loss. Defaults to 128.0.
        stage_percentages (list[float] | tuple[float], optional): Per-stage weights for AT loss.
            Defaults to (0.25, 0.25, 0.25, 0.25).

    Returns:
        torch.Tensor: Combined scalar distillation loss.
    """
    if at_gamma > 0: # Do not calculate AT loss if gamma is 0.
        return kd_loss(student_output=student_output, teacher_output=teacher_output, gamma=kd_gamma, T=T) + at_loss(student_stages=student_stages, teacher_stages=teacher_stages, gamma=at_gamma, stage_percentages=stage_percentages)
    else:
        return kd_loss(student_output=student_output, teacher_output=teacher_output, gamma=kd_gamma, T=T)
