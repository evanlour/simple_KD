import torch

from .functional import ssp_loss


class SSPLoss(torch.nn.Module):
    def __init__(self, gamma: float = 64.0, stage_percentages: tuple[float, ...] | list[float] = (0.25, 0.25, 0.25, 0.25), sort_channels: bool = True):
        """Simple wrapper class for the SSP loss function

        Args:
            gamma (float, optional): Gamma multiplier of the method. Defaults to 64.0.
            stage_percentages (tuple[float] | list[float], optional): Tuple of floats representing each stage's contribution. Defaults to (0.25, 0.25, 0.25, 0.25).
            sort_channels (bool, optional): Toggle true if the results of the teacher and student are sorted before comparing.
        """
        super().__init__()
        self.gamma = gamma
        self.stage_percentages = stage_percentages
        self.sort_channels = sort_channels
    def forward(self, student_stages, teacher_stages):
        return ssp_loss(student_stages=student_stages, teacher_stages=teacher_stages, gamma=self.gamma, stage_percentages=self.stage_percentages, sort_channels=self.sort_channels)
