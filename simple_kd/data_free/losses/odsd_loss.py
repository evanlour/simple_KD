import torch

from .functional import odsd_loss


class ODSDLoss(torch.nn.Module):
    """Wrapper module for the ODSD loss.

    The method is proposed in: Sampling to Distill: Knowledge Transfer from Open-World Data (https://arxiv.org/pdf/2307.16601v2)

    Args:
        kd_temperature (float, optional): Softmax temperature for KD. Defaults to 4.0.
        kd_modifier (float, optional): Multiplier for KD loss. Defaults to 1.0.
        spatial_modifier (float, optional): Multiplier for spatial mapping loss. Defaults to 4.0.
        relational_modifier (float, optional): Multiplier for Barlow Twins relational loss. Defaults to 4.0.
        huber_delta (float, optional): Delta parameter for Huber loss in spatial mapping. Defaults to 1.0.
        tau (float, optional): Constrastive relational loss temperature. Defaults to 0.5.
    """

    def __init__(
        self,
        kd_temperature: float = 4.0,
        kd_modifier: float = 1.0,
        spatial_modifier: float = 4.0,
        relational_modifier: float = 4.0,
        huber_delta: float = 1.0,
        tau: float = 1.0,
    ):
        super().__init__()
        self.kd_temperature = kd_temperature
        self.kd_modifier = kd_modifier
        self.spatial_modifier = spatial_modifier
        self.relational_modifier = relational_modifier
        self.tau = tau
        self.huber_delta = huber_delta
        self.last_loss_components: dict[str, torch.Tensor] = {}

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_basic_output: torch.Tensor,
        teacher_basic_output: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes the ODSD loss.

        Args:
            student_output (torch.Tensor): Output logits from student on augmented images.
            teacher_output (torch.Tensor): Output logits from teacher on augmented images.
            student_basic_output (torch.Tensor): Output logits from student on basic images.
            teacher_basic_output (torch.Tensor): Output logits from teacher on basic images.
            return_dict (bool, optional): If True, returns (total_loss, loss_components_dict).
                Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]: The scalar loss,
                or a tuple of scalar loss and loss dictionary.
        """
        total_loss, loss_dict = odsd_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            student_basic_output=student_basic_output,
            teacher_basic_output=teacher_basic_output,
            kd_temperature=self.kd_temperature,
            kd_modifier=self.kd_modifier,
            spatial_modifier=self.spatial_modifier,
            relational_modifier=self.relational_modifier,
            tau=self.tau,
            huber_delta=self.huber_delta,
        )
        self.last_loss_components = loss_dict
        if return_dict:
            return total_loss, loss_dict
        return total_loss
