import torch

from .functional import dkd_loss


class DKDLoss(torch.nn.Module):
    def __init__(self, T: float = 4.0, tckd_gamma: float = 1.0, nckd_gamma: float = 8.0):
        """Simple wrapper class for the DKD loss function.

        Args:
        T (float, optional): The distillation temperature. Defaults to 4.0.
        tckd_gamma (float, optional): The TCKD gamma AKA the matching label knowledge distillation. Defaults to 1.0.
        nckd_gamma (float, optional): The NCKD gamma AKA the non matching label knowledge distillation. Defaults to 8.0.
        """
        super().__init__()
        self.T = T
        self.tckd_gamma = tckd_gamma
        self.nckd_gamma = nckd_gamma

    def forward(self, student_output, teacher_output, labels):
        return dkd_loss(student_output=student_output, teacher_output=teacher_output, labels=labels, T=self.T, tckd_gamma=self.tckd_gamma, nckd_gamma=self.nckd_gamma)
