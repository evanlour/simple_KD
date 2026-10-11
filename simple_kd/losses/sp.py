import torch

from .functional import sp_loss


class SPLoss(torch.nn.Module):
    def __init__(self, gamma: float = 64.0, stage_percentages: tuple[float, ...] | list[float] = (0.25, 0.25, 0.25, 0.25)):
        """Simple wrapper class for the SP loss function

        Args:
            gamma (float, optional): Gamma multiplier of the method. Defaults to 64.0.
            stage_percentages (tuple[float] | list[float], optional): Tuple of floats representing each stage's contribution. Defaults to (0.25, 0.25, 0.25, 0.25).
        """
        super().__init__()
        self.gamma = gamma
        self.stage_percentages = stage_percentages
    def forward(self, student_stages, teacher_stages):
        return sp_loss(student_stages=student_stages, teacher_stages=teacher_stages, gamma=self.gamma, stage_percentages=self.stage_percentages)
