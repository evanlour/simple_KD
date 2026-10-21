import torch

from .functional import pkt_loss


class PKTLoss(torch.nn.Module):
    """Wrapper module for Probabilistic Knowledge Transfer (PKT) loss.

    Models the probability distribution of sample interactions in the feature space
    using cosine similarities and minimizes the KL divergence between student and teacher distributions.

    The original paper Learning Deep Representations with Probabilistic Knowledge Transfer can be found at https://arxiv.org/pdf/1803.10837.

    Code implementation was mostly extracted from the Knowledge Distillation Zoo repository https://github.com/AberHu/Knowledge-Distillation-Zoo
    along with the mdistiller repository at https://github.com/megvii-research/mdistiller.

    Args:
        gamma (float, optional): Loss multiplier for the method. Defaults to 256.0.
        eps (float, optional): Epsilon to prevent division by zero in normalization. Defaults to 1e-6.
        stage_percentages (list[float] | tuple[float, ...], optional): Relative weights for each stage. Defaults to (0.25, 0.25, 0.25, 0.25).
    """
    def __init__(self, gamma: float = 256.0, eps: float = 1e-6, stage_percentages: list[float] | tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.stage_percentages = stage_percentages

    def forward(self, student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...], teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Computes PKT loss between student and teacher outputs.

        Args:
            student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from student model.
            teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from teacher model.

        Returns:
            torch.Tensor: Scalar PKT loss.
        """
        return pkt_loss(student_stages=student_stages, teacher_stages=teacher_stages, gamma=self.gamma, eps=self.eps, stage_percentages=self.stage_percentages)
