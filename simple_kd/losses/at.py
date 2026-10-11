import torch

from .functional import at_loss


class ATLoss(torch.nn.Module):
    def __init__(self, power: float = 2.0, gamma: float = 64.0, stage_percentages: tuple[float, ...] | list[float]= (0.25, 0.25, 0.25, 0.25)):
        """Simple wrapper class for the AT loss function.

        Args:
            power (float, optional): The power the stages are elevated at to retrieve the attention map. Defaults to 2.0.
            gamma (float, optional): Gamma multiplier of the method. Defaults to 64.0.
            stage_percentages (tuple[float] | list[float], optional): Tuple of floats representing each stage's contribution. Defaults to (0.25, 0.25, 0.25, 0.25).
        """
        super().__init__()
        self.power = power
        self.gamma = gamma
        self.stage_percentages = stage_percentages
    def forward(self, student_stages, teacher_stages):
        return at_loss(student_stages=student_stages, teacher_stages=teacher_stages, power=self.power, gamma=self.gamma, stage_percentages=self.stage_percentages)
