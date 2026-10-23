import torch

from .functional import dafl_loss


class DAFLLoss(torch.nn.Module):
    """Wrapper module for the Data-Free Learning (DAFL) generator loss.

    Combines three intrinsic objectives to train a generator without training data:
    1. Feature Activation Loss: Maximizes intermediate feature responses in the teacher network.
    2. One-Hot Loss: Encourages high classification confidence (low individual sample entropy).
    3. Information Entropy Loss: Maximizes the Shannon entropy across batch-averaged class predictions
       to enforce diversity and prevent class collapse.

    The original paper can be found at https://arxiv.org/pdf/1904.01186 (Chen et al., ICCV 2019).
    The original code implementation can be found at: https://github.com/autogyro/DAFL.

    Args:
        feature_gamma (float, optional): Multiplier for the teacher feature activation loss. Defaults to 1.0.
        one_hot_gamma (float, optional): Multiplier for the teacher one-hot cross-entropy loss. Defaults to 1.0.
        info_entropy_gamma (float, optional): Multiplier for the batch information entropy loss. Defaults to 128.0.
    """

    def __init__(
        self,
        feature_gamma: float = 1.0,
        one_hot_gamma: float = 1.0,
        info_entropy_gamma: float = 128.0,
    ):
        super().__init__()
        self.feature_gamma = feature_gamma
        self.one_hot_gamma = one_hot_gamma
        self.info_entropy_gamma = info_entropy_gamma

    def forward(
        self,
        teacher_output: torch.Tensor,
        teacher_stages: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the combined DAFL loss for the generator.

        Args:
            teacher_output (torch.Tensor): Logits from the teacher network for generated images.
            teacher_stages (list[torch.Tensor] | torch.Tensor): Intermediate feature activation tensors from teacher hooks.

        Returns:
            torch.Tensor: The total scalar generator loss.
        """
        return dafl_loss(
            teacher_output=teacher_output,
            teacher_stages=teacher_stages,
            info_entropy_gamma=self.info_entropy_gamma,
            one_hot_gamma=self.one_hot_gamma,
            feature_gamma=self.feature_gamma,
        )
