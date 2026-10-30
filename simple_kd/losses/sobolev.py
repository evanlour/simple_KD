import torch

from .functional import sobolev_loss


class SobolevLoss(torch.nn.Module):
    """Wrapper module for Sobolev loss.

    The original paper Sobolev Training for Neural Networks can be found at https://arxiv.org/pdf/1706.04859.

    Code implementation is mostly extracted from The mdistiller repository at https://github.com/megvii-research/mdistiller.

    Args:
        gamma (float, optional): The gamma the loss will be multiplied with. Defaults to 256.0.
    """

    def __init__(self, gamma: float = 256.0):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the DKD loss.

        Args:
            student_output (torch.Tensor): Output logits from the student model.
            teacher_output (torch.Tensor): Output logits from the teacher model.
            labels (torch.Tensor): Ground truth target labels.
            images (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The decoupled distillation loss.
        """
        return sobolev_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            labels=labels,
            images=images,
            gamma=self.gamma,
        )
