import torch

from .functional import kd_loss


class KDLoss(torch.nn.Module):
    """Wrapper module for classical Knowledge Distillation (KD) loss.

    Computes Kullback-Leibler (KL) divergence between softened student and teacher probability distributions.

    The original paper is "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015, https://arxiv.org/abs/1503.02531).

    Args:
        T (float, optional): The distillation temperature. Defaults to 4.0.
        gamma (float, optional): Multiplier/weight for the distillation loss. Defaults to 1.0.
    """
    def __init__(self, T: float = 4.0, gamma: float = 1.0):
        super().__init__()
        self.T = T
        self.gamma = gamma

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """Computes the KD loss between student and teacher logits.

        Args:
            student_output (torch.Tensor): Output logits from the student model.
            teacher_output (torch.Tensor): Output logits from the teacher model.

        Returns:
            torch.Tensor: The distillation loss.
        """
        return kd_loss(student_output=student_output, teacher_output=teacher_output, T=self.T, gamma=self.gamma)
