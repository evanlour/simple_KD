import torch

from .functional import pkt_loss


class PKTLoss(torch.nn.Module):
    """Wrapper module for Probabilistic Knowledge Transfer (PKT) loss.

    Models the probability distribution of sample interactions in the feature space
    using cosine similarities and minimizes the KL divergence between student and teacher distributions.

    The original paper is "Learning Deep Representations with Probabilistic Knowledge Transfer"
    (Passalis & Tefas, ECCV 2018, https://arxiv.org/abs/1803.10837).
    The original paper is "Relational Knowledge Distillation" (Park et al., CVPR 2019, https://arxiv.org/abs/1904.05068).
    For all non data free methods the mdistiller https://github.com/megvii-research/mdistiller and the Knowledge distillation zoo https://github.com/AberHu/Knowledge-Distillation-Zoo
    were used as references.

    Args:
        gamma (float, optional): Loss multiplier for the method. Defaults to 256.0.
        eps (float, optional): Epsilon to prevent division by zero in normalization. Defaults to 1e-6.
    """
    def __init__(self, gamma: float = 256.0, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """Computes PKT loss between student and teacher outputs.

        Args:
            student_output (torch.Tensor): Feature representations from student model.
            teacher_output (torch.Tensor): Feature representations from teacher model.

        Returns:
            torch.Tensor: Scalar PKT loss.
        """
        return pkt_loss(student_output=student_output, teacher_output=teacher_output, gamma=self.gamma, eps=self.eps)
