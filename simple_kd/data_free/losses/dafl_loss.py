import torch
from .functional import dafl_loss

class DAFLLoss(torch.nn.Module):
    def __init__(self, feature_gamma: float=1.0, one_hot_gamma: float=1.0, info_entropy_gamma: float=128.0):
        super(DAFLLoss, self).__init__()
        self.feature_gamma = feature_gamma
        self.one_hot_gamma = one_hot_gamma
        self.info_entropy_gamma = info_entropy_gamma

    def forward(self,
                teacher_output: torch.Tensor,
                teacher_stages: torch.Tensor,):
        return dafl_loss(teacher_output=teacher_output, teacher_stages=teacher_stages,
                        info_entropy_gamma=self.info_entropy_gamma, one_hot_gamma=self.one_hot_gamma, feature_gamma=self.feature_gamma)
