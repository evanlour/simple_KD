import torch

from .functional import sp_loss


class SPLoss(torch.nn.Module):
    """Wrapper module for Similarity-Preserving (SP) knowledge distillation loss.

    Preserves pairwise activation similarities of input samples across network layers.


    The original paper is "Similarity-Preserving Knowledge Distillation" (Tung & Mori, ICCV 2019, https://arxiv.org/abs/1907.09682).

    Args:
        gamma (float, optional): Multiplier for the final loss value. Defaults to 64.0.
        stage_percentages (tuple[float, ...] | list[float], optional): Relative weights for each network stage. Defaults to (0.25, 0.25, 0.25, 0.25).
    """
    def __init__(self, gamma: float = 64.0, stage_percentages: tuple[float, ...] | list[float] = (0.25, 0.25, 0.25, 0.25)):
        super().__init__()
        self.gamma = gamma
        self.stage_percentages = stage_percentages

    def forward(self, student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...], teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Computes Similarity-Preserving loss across intermediate stages.

        Args:
            student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from student model.
            teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from teacher model.

        Returns:
            torch.Tensor: Scalar similarity preservation loss.
        """
        return sp_loss(student_stages=student_stages, teacher_stages=teacher_stages, gamma=self.gamma, stage_percentages=self.stage_percentages)
