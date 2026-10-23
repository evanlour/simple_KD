import torch


def dafl_loss(
    teacher_output: torch.Tensor,
    teacher_stages: torch.Tensor,
    feature_gamma: float = 1.0,
    one_hot_gamma: float = 1.0,
    info_entropy_gamma: float = 128.0,
) -> torch.Tensor:
    """Computes the Data-Free Learning (DAFL) generator loss.

     Combines three intrinsic objectives to train a generator without training data:
     1. Feature Activation Loss: Maximizes intermediate feature responses in the teacher network.
     2. One-Hot Loss: Encourages high classification confidence (low individual sample entropy).
     3. Information Entropy Loss: Maximizes the Shannon entropy across batch-averaged class predictions
        to enforce diversity and prevent class collapse.

    The original paper can be found at https://arxiv.org/pdf/1904.01186 (Chen et al., ICCV 2019).
    The original code implementation can be found at: https://github.com/autogyro/DAFL.

    Args:
        teacher_output (torch.Tensor): Teacher model logits for the generated images batch.
        teacher_stages (list[torch.Tensor] | torch.Tensor): Intermediate feature activation tensors from teacher hooks.
        feature_gamma (float, optional): Multiplier for feature activation loss. Defaults to 1.0.
        one_hot_gamma (float, optional): Multiplier for one-hot cross-entropy loss. Defaults to 1.0.
        info_entropy_gamma (float, optional): Multiplier for batch information entropy loss. Defaults to 128.0.

    Returns:
        torch.Tensor: Combined scalar loss for the generator.
    """
    feature_loss = -sum([f.abs().mean() for f in teacher_stages]) * feature_gamma
    targets = torch.argmax(teacher_output, dim=1).detach()
    one_hot_loss = (
        torch.nn.functional.cross_entropy(teacher_output, targets) * one_hot_gamma
    )
    teacher_softmax = torch.nn.functional.softmax(teacher_output, dim=1).mean(dim=0)
    info_entropy_loss = (
        teacher_softmax * torch.log(teacher_softmax + 1e-6)
    ).sum() * info_entropy_gamma
    return feature_loss + one_hot_loss + info_entropy_loss
