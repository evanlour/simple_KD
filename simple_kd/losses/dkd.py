import torch

from .functional import dkd_loss


class DKDLoss(torch.nn.Module):
    """Wrapper module for Decoupled Knowledge Distillation (DKD) loss.

    Decouples classical KD into Target Class Knowledge Distillation (TCKD) and
    Non-Target Class Knowledge Distillation (NCKD):
        L_DKD = tckd_gamma * L_TCKD + nckd_gamma * L_NCKD

    The original paper is "Decoupled Knowledge Distillation" (Zhao et al., CVPR 2022, https://arxiv.org/abs/2203.08679).

    Args:
        T (float, optional): Distillation temperature. Defaults to 4.0.
        tckd_gamma (float, optional): Multiplier for Target Class Knowledge Distillation. Defaults to 1.0.
        nckd_gamma (float, optional): Multiplier for Non-target Class Knowledge Distillation. Defaults to 8.0.
    """
    def __init__(self, T: float = 4.0, tckd_gamma: float = 1.0, nckd_gamma: float = 8.0):
        super().__init__()
        self.T = T
        self.tckd_gamma = tckd_gamma
        self.nckd_gamma = nckd_gamma

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes the DKD loss.

        Args:
            student_output (torch.Tensor): Output logits from the student model.
            teacher_output (torch.Tensor): Output logits from the teacher model.
            labels (torch.Tensor): Ground truth target labels.

        Returns:
            torch.Tensor: The decoupled distillation loss.
        """
        return dkd_loss(student_output=student_output, teacher_output=teacher_output, labels=labels, T=self.T, tckd_gamma=self.tckd_gamma, nckd_gamma=self.nckd_gamma)
