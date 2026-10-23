import torch

from .functional import ssp_loss


class SSPLoss(torch.nn.Module):
    """Wrapper module for Sorted/Simplified Similarity-Preserving (SSP) feature loss.

    Extracts spatial pooled feature representations across stages, optionally sorts channel
    magnitudes to align dominant representations, and computes MSE loss between matched feature widths.

    This is a method i created based on the Fitnets method (Romero et al, 2014), https://arxiv.org/abs/1412.6550.
    The original code can be found at https://github.com/adri-romsor/FitNets.

    Args:
        gamma (float, optional): Multiplier for the final loss value. Defaults to 64.0.
        stage_percentages (tuple[float, ...] | list[float], optional): Relative weights for each network stage. Defaults to (0.25, 0.25, 0.25, 0.25).
        sort_channels (bool, optional): Whether to sort channel activations in descending order before comparison. Defaults to True.
    """

    def __init__(
        self,
        gamma: float = 64.0,
        stage_percentages: tuple[float, ...] | list[float] = (0.25, 0.25, 0.25, 0.25),
        sort_channels: bool = True,
    ):
        super().__init__()
        self.gamma = gamma
        self.stage_percentages = stage_percentages
        self.sort_channels = sort_channels

    def forward(
        self,
        student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Computes SSP loss across intermediate stages.

        Args:
            student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from student model.
            teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from teacher model.

        Returns:
            torch.Tensor: Scalar SSP loss.
        """
        return ssp_loss(
            student_stages=student_stages,
            teacher_stages=teacher_stages,
            gamma=self.gamma,
            stage_percentages=self.stage_percentages,
            sort_channels=self.sort_channels,
        )
