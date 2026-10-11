import torch

from .functional import skeptical_loss


class SkepticalClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        """An auxiliary classifier used to match intermediate model activations. Used by the skeptical student method.

        Args:
            in_channels (int): The input, AKA the size of the stage.
            num_classes (int): The output classes.
        """
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class SkepticalLoss(torch.nn.Module):
    def __init__(self, T: float = 4.0, kd_gamma: float = 0.5, in_kd_gamma: float = 1.0):
        """Simple wrapper class for the Skeptical loss function

        Args:
            T (float, optional): The distillation temperature. Defaults to 4.0.
            kd_gamma (float, optional): The traditional KD part gamma. Defaults to 0.5.
            in_kd_gamma (float, optional): The esoterical KD inside the student model gamma. Defaults to 1.0.
        """
        super().__init__()
        self.T = T
        self.kd_gamma = kd_gamma
        self.in_kd_gamma = in_kd_gamma
    def forward(self, student_output, teacher_output, student_stages, aux_classifier):
        return skeptical_loss(student_output=student_output, teacher_output=teacher_output, student_stages=student_stages, aux_classifier=aux_classifier, T=self.T, kd_gamma=self.kd_gamma, in_kd_gamma=self.in_kd_gamma)
