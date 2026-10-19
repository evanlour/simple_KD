import torch


def dafl_loss(teacher_output: torch.Tensor, teacher_stages: torch.Tensor,
              feature_gamma: float=1.0, one_hot_gamma: float=1.0, info_entropy_gamma: float=128.0) -> torch.Tensor:
    """Computes the Data-Free Learning (DAFL) generator loss.

    The DAFL generator loss combines three terms to guide synthesis without training data:
    1. Feature Activation Loss: Maximizes intermediate feature responses in the teacher network:
       L_feat = -sum(mean(|f|)) * feature_gamma
    2. One-Hot Loss: Cross-entropy with teacher's predicted pseudo-label to enforce sharp class identity:
       L_oh = CrossEntropy(teacher_output, argmax(teacher_output)) * one_hot_gamma
    3. Information Entropy Loss: Shannon entropy over the batch-averaged class distribution to prevent mode collapse:
       L_ie = sum(p_bar * log(p_bar)) * info_entropy_gamma

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
    one_hot_loss = torch.nn.functional.cross_entropy(teacher_output, targets) * one_hot_gamma
    teacher_softmax = torch.nn.functional.softmax(teacher_output, dim=1).mean(dim=0)
    info_entropy_loss = (teacher_softmax * torch.log(teacher_softmax + 1e-6)).sum() * info_entropy_gamma
    return feature_loss + one_hot_loss + info_entropy_loss
