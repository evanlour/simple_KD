import torch
import torch.nn.functional as F


def _get_sp_feats(stage: torch.Tensor) -> torch.Tensor:
    """Helper function that returns the matrix similarities."""
    stage_flat = stage.view(stage.size(0), -1)
    feats = torch.mm(stage_flat, stage_flat.t())
    feats = F.normalize(feats, p=2, dim=1)
    return feats

def sp_loss(student_stages: torch.Tensor,
           teacher_stages: torch.Tensor,
           stage_percentages: tuple[float, ...] | list[float] = (0.25, 0.25, 0.25, 0.25),
           gamma: float = 64.0) -> torch.Tensor:
    """Code implementation of the Similarity Preservation (SP) method.

    The original paper can be found at https://arxiv.org/pdf/1907.09682.

    Code implementation was originally from https://github.com/AberHu/Knowledge-Distillation-Zoo.

    Similar code can be found at https://github.com/megvii-research/mdistiller/tree/master.

    Args:
        student_stages (list[torch.Tensor]): List of intermediate activations for the student model.
        teacher_stages (list[torch.Tensor]): List of intermediate activations for the teacher model.
        stage_percentages (tuple[float] | list[float], optional): Tuple of floats representing each stage's contribution. Defaults to (0.25, 0.25, 0.25, 0.25).
        gamma (float, optional): Multiplies the final loss value by gamma. Defaults to 64.0.

    Returns:
        torch.Tensor: The distillation loss.
    """
    loss = 0
    for teacher_stage, student_stage, stage_percentage in zip(teacher_stages, student_stages, stage_percentages):
        teacher_feats = _get_sp_feats(teacher_stage)
        student_feats = _get_sp_feats(student_stage)
        loss += F.mse_loss(student_feats, teacher_feats) * stage_percentage
    return loss * gamma # type: ignore
