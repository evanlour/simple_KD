import torch

from .functional import dfad_loss


class DFADLoss(torch.nn.Module):
    """Wrapper module for the Data-Free Adversarial Distillation (DFAD) loss.

    Computes L1 discrepancy between student and teacher outputs.
    - Used with sign > 0 for the student model (minimizing discrepancy).
    - Used with sign < 0 for the generator network (maximizing discrepancy).

    The original paper is "Data-Free Adversarial Distillation" (Fang et al., CVPR 2020), https://arxiv.org/abs/1912.11006.
    The original code can be found at https://github.com/VainF/Data-Free-Adversarial-Distillation.

    Args:
        sign (float, optional): The sign and multiplier of the result (+1.0 for student, -1.0 for generator). Defaults to 1.0.
        use_log1p (bool, optional): Whether to apply log1p to the L1 loss (ADA protocol). Defaults to False.
    """
    def __init__(self, sign: float=1.0, use_log1p:bool=False):
        super().__init__()
        self.sign = sign
        self.use_log1p = use_log1p

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """Computes the signed DFAD loss.

        Args:
            student_output (torch.Tensor): Output logits from the student model.
            teacher_output (torch.Tensor): Output logits from the teacher model.

        Returns:
            torch.Tensor: Scalar discrepancy loss scaled by sign.
        """
        if self.use_log1p:
            return self.sign * torch.log1p(dfad_loss(student_output=student_output, teacher_output=teacher_output))
        else:
            return self.sign * dfad_loss(student_output=student_output, teacher_output=teacher_output)
