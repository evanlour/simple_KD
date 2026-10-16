import torch

from .functional import rkd_angle_loss, rkd_dist_loss


class RKDAngleLoss(torch.nn.Module):
    def __init__(self, gamma: float = 64.0, stage_percentages: list[float] | tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)):
        """Simple wrapper class for the RKD angle loss function.

        Args:
        gamma (float, optional): Multiplies the final loss value by gamma. Defaults to 64.0.
        stage_percentages (list[float] | tuple[float], optional): Tuple of floats representing each stage's contribution. Defaults to (0.25, 0.25, 0.25, 0.25).
        """
        super().__init__()
        self.stage_percentages = stage_percentages
        self.gamma = gamma
    def forward(self, student_stages, teacher_stages):
        return rkd_angle_loss(student_stages=student_stages, teacher_stages=teacher_stages, gamma=self.gamma, stage_percentages=self.stage_percentages)

class RKDDistLoss(torch.nn.Module):
    def __init__(self, gamma: float = 64.0, stage_percentages: list[float] | tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)):
        """Simple wrapper class for the RKD dist loss function.

        Args:
        gamma (float, optional): Multiplies the final loss value by gamma. Defaults to 64.0.
        stage_percentages (list[float] | tuple[float], optional): Tuple of floats representing each stage's contribution. Defaults to (0.25, 0.25, 0.25, 0.25).
        """
        super().__init__()
        self.stage_percentages = stage_percentages
        self.gamma = gamma
    def forward(self, student_stages, teacher_stages):
        return rkd_dist_loss(student_stages=student_stages, teacher_stages=teacher_stages, gamma=self.gamma, stage_percentages=self.stage_percentages)
