import torch
from .functional import zskt_loss

class ZSKTLoss(torch.nn.Module):
    def __init__(self, T: float=4.0, kd_gamma: float=1.0, at_gamma: float=128.0,
              stage_percentages: list[float] | tuple[float] = [0.25, 0.25, 0.25, 0.25]):
        super(ZSKTLoss, self).__init__()
        self.T = T
        self.kd_gamma = kd_gamma
        self.at_gamma = at_gamma
        self.stage_percentages = stage_percentages

    def forward(self, student_output: torch.Tensor,
                teacher_output: torch.Tensor,
                student_stages: torch.Tensor,
                teacher_stages: torch.Tensor,):
        return zskt_loss(student_output=student_output, teacher_output=teacher_output, student_stages=student_stages, teacher_stages=teacher_stages, T=self.T, kd_gamma=self.kd_gamma, at_gamma=self.at_gamma, stage_percentages=self.stage_percentages)
