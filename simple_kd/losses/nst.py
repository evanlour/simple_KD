import torch

from .functional import nst_loss


class NSTLoss(torch.nn.Module):
    """Wrapper module for Neuron Selectivity Transfer (NST) loss.

    Computes the neuron selectivity transfer loss between student and teacher feature maps.

    The original paper is "Like What You Like: Knowledge Distill via Neuron Selectivity Transfer", https://arxiv.org/pdf/1707.01219.

    Args:
        gamma (float, optional): The gamma of the loss. Defaults to 256.0.
        stage_percentages (list[float] | tuple[float], optional): Per-stage weight fractions for NST loss.
            Defaults to (0.1, 0.1, 0.3, 0.6).
    """

    def __init__(
        self,
        gamma: float = 256.0,
        stage_percentages: tuple[float, ...] | list[float] = (0.1, 0.1, 0.3, 0.6),
    ):
        super().__init__()
        self.gamma = gamma
        self.stage_percentages = stage_percentages

    def forward(
        self,
        student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Computes the NST loss.

        Args:
            student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate feature maps from student model.
            teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate feature maps from teacher model.

        Returns:
            torch.Tensor: The NST loss.
        """
        return nst_loss(
            student_stages=student_stages,
            teacher_stages=teacher_stages,
            stage_percentages=self.stage_percentages,
            gamma=self.gamma
        )
