import torch

from .functional import skeptical_loss


class SkepticalClassifier(torch.nn.Module):
    """An auxiliary classifier used to project intermediate student activations into class logits.

    Used by the Skeptical Student distillation method to extract intermediate predictions.

    Args:
        in_channels (int): Input channel dimension of the intermediate feature stage.
        num_classes (int): Number of target classification classes.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects feature map to class logits.

        Args:
            x (torch.Tensor): Intermediate feature map.

        Returns:
            torch.Tensor: Logits vector of shape [batch_size, num_classes].
        """
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class SkepticalLoss(torch.nn.Module):
    """Wrapper module for Skeptical Student distillation loss.

    Trains students to learn robust representations when distilled from imperfect or poisoned
    teachers ("nasty teachers") by combining output KD with internal self-distillation.

    The original paper can be found here https://openreview.net/pdf?id=Laz0L5tjml.

    The original code implementation by the authors can be found here https://github.com/ksouvik52/Skeptical2021.

    Args:
        T (float, optional): The distillation temperature. Defaults to 4.0.
        kd_gamma (float, optional): Weight multiplier for standard KD loss. Defaults to 0.5.
        in_kd_gamma (float, optional): Weight multiplier for intermediate internal KD loss. Defaults to 1.0.
    """
    def __init__(self, T: float = 4.0, kd_gamma: float = 0.5, in_kd_gamma: float = 1.0):
        super().__init__()
        self.T = T
        self.kd_gamma = kd_gamma
        self.in_kd_gamma = in_kd_gamma

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor,
                student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
                aux_classifier: torch.nn.Module) -> torch.Tensor:
        """Computes the skeptical distillation loss.

        Args:
            student_output (torch.Tensor): Logits from the student model.
            teacher_output (torch.Tensor): Logits from the teacher model.
            student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): Intermediate activations from student model.
            aux_classifier (torch.nn.Module): Auxiliary classifier module projecting intermediate activations.

        Returns:
            torch.Tensor: Combined skeptical distillation loss.
        """
        return skeptical_loss(student_output=student_output, teacher_output=teacher_output, student_stages=student_stages, aux_classifier=aux_classifier, T=self.T, kd_gamma=self.kd_gamma, in_kd_gamma=self.in_kd_gamma)
