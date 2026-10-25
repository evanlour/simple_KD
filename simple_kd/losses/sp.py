import torch

from .functional import sp_loss


class SPLoss(torch.nn.Module):
    """Wrapper module for Similarity-Preserving (SP) knowledge distillation loss.

    Preserves pairwise activation similarities of input samples across network layers.

    The original paper can be found at https://arxiv.org/pdf/1907.09682.

    Code implementation was originally from https://github.com/AberHu/Knowledge-Distillation-Zoo.

    Similar code can be found at https://github.com/megvii-research/mdistiller/tree/master.

    Args:
        gamma (float, optional): Multiplier for the final loss value. Defaults to 512.0.
        stage_percentages (tuple[float, ...] | list[float], optional): Relative weights for each network stage. Defaults to (0.1, 0.1, 0.2, 0.6).
    """

    def __init__(
        self,
        gamma: float = 512.0,
        stage_percentages: tuple[float, ...] | list[float] = (0.1, 0.1, 0.2, 0.6),
    ):
        super().__init__()
        self.gamma = gamma
        self.stage_percentages = stage_percentages

    def forward(
        self,
        student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Computes Similarity-Preserving loss across intermediate stages.

        Args:
            student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from student model.
            teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from teacher model.

        Returns:
            torch.Tensor: Scalar similarity preservation loss.
        """
        return sp_loss(
            student_stages=student_stages,
            teacher_stages=teacher_stages,
            gamma=self.gamma,
            stage_percentages=self.stage_percentages,
        )
