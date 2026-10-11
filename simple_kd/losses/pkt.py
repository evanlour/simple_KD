import torch

from .functional import pkt_loss


class PKTLoss(torch.nn.Module):
    def __init__(self, gamma: float = 256.0, eps: float = 1e-6):
        """Simple wrapper class for the KD loss function.

        Args:
            gamma (float, optional): Loss multiplier for the method. Defaults to 256.0.
            eps (float, optional): Prevents division by zero. Defaults to 1e-6.
        """
        super().__init__()
        self.gamma = gamma
        self.eps = eps
    def forward(self, student_output, teacher_output):
        return pkt_loss(student_output=student_output, teacher_output=teacher_output, gamma=self.gamma, eps=self.eps)
