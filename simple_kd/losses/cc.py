import torch

from .functional import cc_loss


class CCLoss(torch.nn.Module):
    """Wrapper class for the correlation congruence method.

    Moves instance and correlation level information from the teacher to the student.

    Method is from the paper Correlation Congruence for Knowledge Distillation (https://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf).

    Args:
        gamma (float, optional): Multiplier for the final loss value. Defaults to 1.0.
        stage_percentages (tuple[float, ...] | list[float], optional): Relative weights for each network stage. Defaults to (0.1, 0.1, 0.2, 0.6).
    """

    def __init__(
        self,
        gamma: float = 1.0,
        stage_percentages: tuple[float, ...] | list[float] = (0.1, 0.1, 0.2, 0.6),
    ):
        super().__init__()
        self.gamma = gamma
        self.stage_percentages = stage_percentages

    def forward(
        self,
        student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Computes SSP loss across intermediate stages.

        Args:
            student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from student model.
            teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from teacher model.

        Returns:
            torch.Tensor: Scalar CC loss.
        """
        return cc_loss(
            student_stages=student_stages,
            teacher_stages=teacher_stages,
            gamma=self.gamma,
            stage_percentages=self.stage_percentages,
        )
