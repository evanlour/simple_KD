import torch

from .functional import rkd_angle_loss, rkd_dist_loss


class RKDAngleLoss(torch.nn.Module):
    """Wrapper module for Relational Knowledge Distillation (RKD): Angle-wise loss.

    Penalizes angular differences of sample triplets in the embedding space.

    The original paper Relational Knowledge Distillation can be found at https://arxiv.org/pdf/1904.05068.

    Code implementation was originally from https://github.com/AberHu/Knowledge-Distillation-Zoo.

    Similar code can be found at https://github.com/megvii-research/mdistiller/tree/master.

    Args:
        gamma (float, optional): Multiplier for the final loss value. Defaults to 64.0.
        stage_percentages (list[float] | tuple[float, ...], optional): Relative weights for each stage. Defaults to (0.1, 0.1, 0.2, 0.6).
    """

    def __init__(
        self,
        gamma: float = 64.0,
        stage_percentages: list[float] | tuple[float, ...] = (0.1, 0.1, 0.2, 0.6),
    ):
        super().__init__()
        self.stage_percentages = stage_percentages
        self.gamma = gamma

    def forward(
        self,
        student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Computes RKD angle loss across stages.

        Args:
            student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from student model.
            teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from teacher model.

        Returns:
            torch.Tensor: Scalar angle-wise relational loss.
        """
        return rkd_angle_loss(
            student_stages=student_stages,
            teacher_stages=teacher_stages,
            gamma=self.gamma,
            stage_percentages=self.stage_percentages,
        )


class RKDDistLoss(torch.nn.Module):
    """Wrapper module for Relational Knowledge Distillation (RKD): Distance-wise loss.

    Penalizes differences in normalized pairwise Euclidean distances between sample pairs.

    The original paper Relational Knowledge Distillation can be found at https://arxiv.org/pdf/1904.05068.

    Code implementation was originally from https://github.com/AberHu/Knowledge-Distillation-Zoo.

    Similar code can be found at https://github.com/megvii-research/mdistiller/tree/master.

    Args:
        gamma (float, optional): Multiplier for the final loss value. Defaults to 128.0.
        stage_percentages (list[float] | tuple[float, ...], optional): Relative weights for each stage. Defaults to (0.1, 0.1, 0.2, 0.6).
    """

    def __init__(
        self,
        gamma: float = 128.0,
        stage_percentages: list[float] | tuple[float, ...] = (0.1, 0.1, 0.2, 0.6),
    ):
        super().__init__()
        self.stage_percentages = stage_percentages
        self.gamma = gamma

    def forward(
        self,
        student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Computes RKD distance loss across stages.

        Args:
            student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from student model.
            teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from teacher model.

        Returns:
            torch.Tensor: Scalar distance-wise relational loss.
        """
        return rkd_dist_loss(
            student_stages=student_stages,
            teacher_stages=teacher_stages,
            gamma=self.gamma,
            stage_percentages=self.stage_percentages,
        )
