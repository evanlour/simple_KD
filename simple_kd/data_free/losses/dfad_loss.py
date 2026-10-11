import torch
from .functional import dfad_loss

class DFADLoss(torch.nn.Module):
    def __init__(self, sign=1, use_log1p=False):
        """A very simple wrapper class for the dfad_loss method.

        Args:
            sign (int, optional): The sign of the result, also works as gamma. Defaults to 1.
            use_log1p (bool, optional): Use the ada protocol AKA using log_1p on the logits of the generator. Defaults to False.
        """
        super(DFADLoss, self).__init__()
        self.sign = sign
        self.use_log1p = use_log1p

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor):
        if self.use_log1p:
            return self.sign * torch.log1p(dfad_loss(student_output=student_output, teacher_output=teacher_output))
        else:
            return self.sign * dfad_loss(student_output=student_output, teacher_output=teacher_output)
