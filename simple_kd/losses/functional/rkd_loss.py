import torch
import torch.nn.functional as F


def _get_pdist(feat, eps=1e-12):
    """Inner helper function used to get the feature distances from an intermediate activation."""
    feat_square = feat.pow(2).sum(dim=1)
    feat_prod   = torch.mm(feat, feat.t())
    feat_dist   = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

    # Take square root and fix the diagonal
    feat_dist = feat_dist.sqrt().clone()
    feat_dist[range(len(feat)), range(len(feat))] = 0 # The feature cannot have distance from itself
    return feat_dist

def _get_angle_feats(stage: torch.Tensor) -> torch.Tensor:
    """Helper that returns the angle features of the given stage."""
    # We Calculate all possible triplets for the teacher, N x C --> N x N x C
    flat_stage = F.adaptive_avg_pool2d(stage, (1, 1)).view(stage.size(0), -1)
    stage_triplets = (flat_stage.unsqueeze(0) - flat_stage.unsqueeze(1))
    stage_triplets = F.normalize(stage_triplets, p=2, dim=2) # We normalize the distances as we are searching angles
    stage_feats = torch.bmm(stage_triplets, stage_triplets.transpose(1, 2)).view(-1)
    return stage_feats

def _get_distance_feats(stage: torch.Tensor) -> torch.Tensor:
    """Helper that returns the distance features of the given stage."""
    # First we flatten the feats and calculate their distances squared separately
    flat_stage = F.adaptive_avg_pool2d(stage, (1, 1)).view(stage.size(0), -1)
    stage_dist_feats = _get_pdist(flat_stage)
    mean_stage_dist_feats = stage_dist_feats[stage_dist_feats > 0].mean()
    stage_dist_feats = stage_dist_feats / mean_stage_dist_feats
    return stage_dist_feats

def rkd_angle_loss(student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
            teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
            stage_percentages: tuple[float, ...] | list[float] = (0.25, 0.25, 0.25, 0.25),
            gamma: float = 64.0) -> torch.Tensor:
    """Code implementation of the RKD method: Angle variant.

    The original paper Relational Knowledge Distillation can be found at https://arxiv.org/pdf/1904.05068.

    Code implementation was originally from https://github.com/AberHu/Knowledge-Distillation-Zoo.

    Similar code can be found at https://github.com/megvii-research/mdistiller/tree/master.

    Args:
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): List of intermediate activations for the student model.
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): List of intermediate activations for the teacher model.
        stage_percentages (list[float] | tuple[float, ...], optional): Tuple of floats representing each stage's contribution. Defaults to (0.25, 0.25, 0.25, 0.25).
        gamma (float, optional): Multiplies the final loss value by gamma. Defaults to 64.0.

    Returns:
        torch.Tensor: The distillation loss.
    """
    loss = 0
    for teacher_stage, student_stage, stage_percentage in zip(teacher_stages, student_stages, stage_percentages):
        teacher_angle_feats = _get_angle_feats(teacher_stage)
        student_angle_feats = _get_angle_feats(student_stage)
        loss += F.smooth_l1_loss(student_angle_feats, teacher_angle_feats) * stage_percentage # huber loss
    return loss * gamma # type: ignore

def rkd_dist_loss(student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
        stage_percentages: tuple[float, ...] | list[float] = (0.25, 0.25, 0.25, 0.25),
        gamma: float = 4.0) -> torch.Tensor:
    """Code implementation of the RKD method: Distance variant.

    The original paper Relational Knowledge Distillation can be found at https://arxiv.org/pdf/1904.05068.

    Code implementation was originally from https://github.com/AberHu/Knowledge-Distillation-Zoo.

    Similar code can be found at https://github.com/megvii-research/mdistiller/tree/master.

    Args:
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): List of intermediate activations for the student model.
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): List of intermediate activations for the teacher model.
        stage_percentages (list[float] | tuple[float, ...], optional): Tuple of floats representing each stage's contribution. Defaults to (0.25, 0.25, 0.25, 0.25).
        gamma (float, optional): Multiplies the final loss value by gamma. Defaults to 64.0.

    Returns:
        torch.Tensor: The distillation loss.
    """
    loss = 0
    for teacher_stage, student_stage, stage_percentage in zip(teacher_stages, student_stages, stage_percentages):
        teacher_dist_feats = _get_distance_feats(teacher_stage)
        student_dist_feats = _get_distance_feats(student_stage)
        loss += F.smooth_l1_loss(student_dist_feats, teacher_dist_feats) * stage_percentage # huber loss
    return loss * gamma # type: ignore
