import torch

from .functional import zskt_loss


class ZSKTLoss(torch.nn.Module):
    """Wrapper module for Zero-Shot Knowledge Transfer (ZSKT) loss.

    Combines output knowledge distillation (KDLoss) with intermediate attention
    transfer (ATLoss) across network stages.
    - Used with sign > 0 for the student model (minimizing discrepancy).
    - Used with sign < 0 for the generator network (maximizing discrepancy).

    The original paper is "Zero-shot Knowledge Transfer via Adversarial Belief Matching"
    (Micaelli & Storkey, NeurIPS 2019, https://arxiv.org/abs/1905.09768). The code is available
    here https://github.com/polo5/ZeroShotKnowledgeTransfer

    Args:
        T (float, optional): Distillation temperature for KD loss. Defaults to 4.0.
        kd_gamma (float, optional): Multiplier for the output KD loss. Defaults to 1.0.
        at_gamma (float, optional): Multiplier for the intermediate Attention Transfer loss. Defaults to 128.0.
        stage_percentages (list[float] | tuple[float], optional): Per-stage weight fractions for AT loss. Defaults to [0.25, 0.25, 0.25, 0.25].
        sign (float, optional): The sign and multiplier of the result (+1.0 for student, -1.0 for generator). Defaults to 1.0.
    """
    def __init__(self, T: float=4.0, kd_gamma: float=1.0, at_gamma: float=128.0,
              stage_percentages: list[float] | tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
              sign: float=1.0):
        super().__init__()
        self.sign = sign
        self.T = T
        self.kd_gamma = kd_gamma
        self.at_gamma = at_gamma
        self.stage_percentages = stage_percentages

    def forward(self, student_output: torch.Tensor,
                teacher_output: torch.Tensor,
                student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
                teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Computes combined ZSKT distillation loss.

        Args:
            student_output (torch.Tensor): Logits from the student model.
            teacher_output (torch.Tensor): Logits from the teacher model.
            student_stages (list[torch.Tensor] | torch.Tensor): Intermediate feature tensors from student hooks.
            teacher_stages (list[torch.Tensor] | torch.Tensor): Intermediate feature tensors from teacher hooks.

        Returns:
            torch.Tensor: Combined scalar loss scaled by sign.
        """
        return self.sign * zskt_loss(student_output=student_output, teacher_output=teacher_output, student_stages=student_stages, teacher_stages=teacher_stages, T=self.T, kd_gamma=self.kd_gamma, at_gamma=self.at_gamma, stage_percentages=self.stage_percentages)
