import torch

def dafl_loss(teacher_output: torch.Tensor, teacher_stages: torch.Tensor,
              feature_gamma: float=1.0, one_hot_gamma: float=1.0, info_entropy_gamma: float=128.0):
    feature_loss = -sum([f.abs().mean() for f in teacher_stages]) * feature_gamma
    targets = torch.argmax(teacher_output, dim=1).detach()
    one_hot_loss = torch.nn.functional.cross_entropy(teacher_output, targets) * one_hot_gamma
    teacher_softmax = torch.nn.functional.softmax(teacher_output, dim=1).mean(dim=0)
    info_entropy_loss = (teacher_softmax * torch.log(teacher_softmax + 1e-6)).sum() * info_entropy_gamma
    return feature_loss + one_hot_loss + info_entropy_loss
