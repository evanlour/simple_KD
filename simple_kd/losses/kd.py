import torch

from .functional import kd_loss


class KDLoss(torch.nn.Module):
    def __init__(self, T: float = 4.0, gamma: float = 1.0):
        """Simple wrapper class for the KD loss function.

        Args:
            T (float, optional): The distillation temperature. Defaults to 4.0.
            gamma (float, optional): Gamma multiplier of the method. Defaults to 1.0.
        """
        super().__init__()
        self.T = T
        self.gamma = gamma
    def forward(self, student_output, teacher_output):
        return kd_loss(student_output=student_output, teacher_output=teacher_output, T=self.T, gamma=self.gamma)
