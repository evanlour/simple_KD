import torch

from .functional import at_loss


class ATLoss(torch.nn.Module):
    """Wrapper module for Attention Transfer (AT) loss.

    The original paper (PAYING MORE ATTENTION TO ATTENTION: IMPROVING THE PERFORMANCE OF CONVOLUTIONAL
    NEURAL NETWORKS VIA ATTENTION TRANSFER) can be found at https://arxiv.org/pdf/1612.03928.

    Code implementation was originally from https://github.com/AberHu/Knowledge-Distillation-Zoo.

    Similar code can be found at https://github.com/megvii-research/mdistiller/tree/master.

    Args:
        power (float, optional): Exponent used to construct attention maps. Defaults to 2.0.
        gamma (float, optional): Multiplier for the final loss value. Defaults to 64.0.
        stage_percentages (tuple[float, ...] | list[float], optional): Relative weights for each network stage. Defaults to (0.25, 0.25, 0.25, 0.25).
    """

    def __init__(
        self,
        power: float = 2.0,
        gamma: float = 64.0,
        stage_percentages: tuple[float, ...] | list[float] = (0.25, 0.25, 0.25, 0.25),
    ):
        super().__init__()
        self.power = power
        self.gamma = gamma
        self.stage_percentages = stage_percentages

    def forward(
        self,
        student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Computes Attention Transfer loss across intermediate stages.

        Args:
            student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from student model.
            teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from teacher model.

        Returns:
            torch.Tensor: Scalar attention transfer loss.
        """
        return at_loss(
            student_stages=student_stages,
            teacher_stages=teacher_stages,
            power=self.power,
            gamma=self.gamma,
            stage_percentages=self.stage_percentages,
        )
