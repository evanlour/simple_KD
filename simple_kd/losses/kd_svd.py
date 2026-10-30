import torch

from .functional import kd_svd_loss


class KD_SVDLoss(torch.nn.Module):
    """Wrapper module for the knowledge distillation using svd method.

    The original paper is Self-supervised Knowledge Distillation Using Singular Value Decomposition
    and can be found at https://arxiv.org/pdf/1807.06819

    Code implementation was originally from https://github.com/sseung0703/SSKD_SVD

    Args:
        stage_percentages (tuple[float] | list[float], optional): Tuple of floats representing each stage's contribution. Defaults to (0.1, 0.1, 0.2, 0.6).
        rank (int, optional): The top ranks of the SVD to choose. Defaults to 3.
        rank_extras (int, optional): The additional top ranks kept from the student feature. Defaults to 3.
        rbf_sigma (float, optional): The number applied to the rbf to smoothen it. Defaults to 8.0.
        gamma (float, optional): Multiplies the final loss value by gamma. Defaults to 32.0.
    """

    def __init__(
        self,
        rank: int = 1,
        rank_extras: int = 2,
        rbf_sigma: float = 8.0,
        gamma: float = 32.0,
        stage_percentages: tuple[float, ...] | list[float] = (0.1, 0.1, 0.2, 0.6),
    ):
        super().__init__()
        self.rank = rank
        self.rank_extras = rank_extras
        self.rbf_sigma = rbf_sigma
        self.gamma = gamma
        self.stage_percentages = stage_percentages

    def forward(
        self,
        student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Computes the KD SVD loss across intermediate stages.

        Args:
            student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from student model.
            teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from teacher model.

        Returns:
            torch.Tensor: Scalar attention transfer loss.
        """
        return kd_svd_loss(
            student_stages=student_stages,
            teacher_stages=teacher_stages,
            rank=self.rank,
            rank_extras=self.rank_extras,
            rbf_sigma=self.rbf_sigma,
            gamma=self.gamma,
            stage_percentages=self.stage_percentages,
        )
