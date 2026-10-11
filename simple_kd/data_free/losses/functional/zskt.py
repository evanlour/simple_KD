import torch
from simple_kd.losses.functional import at_loss, kd_loss

def zskt_loss(student_output: torch.Tensor, teacher_output: torch.Tensor,
              student_stages: torch.Tensor, teacher_stages: torch.Tensor,
              T: float=4.0, kd_gamma: float=1.0, at_gamma: float=128.0,
              stage_percentages: list[float] | tuple[float] = [0.25, 0.25, 0.25, 0.25]):
    return kd_loss(student_output=student_output, teacher_output=teacher_output, gamma=kd_gamma, T=T) + at_loss(student_stages=student_stages, teacher_stages=teacher_stages, gamma=at_gamma, stage_percentages=stage_percentages)
