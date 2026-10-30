import torch
import torch.nn.functional as F

# def _get_svd(feat: torch.Tensor, n: int = 1):
#     x = feat.flatten(1, -2).float()
#     u, s, v = torch.linalg.svd(x, full_matrices=False)
#     v = v.mH
#     u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
#     s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
#     v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
#
#     if n > 0:
#         u = F.normalize(u[:, :, :n], dim=1)
#         s = F.normalize(s[:, :n], dim=1)
#         v = F.normalize(v[:, :, :n], dim=1)
#     return u, s, v


def _align_rsv(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        cosine = torch.matmul(a.mT, b)
        idx = cosine.abs().argmax(dim=1, keepdim=True)
        signs = torch.sgn(cosine.gather(1, idx))
    return a.gather(2, idx.expand(-1, a.shape[1], -1)) * signs, b


def _top_rsv_pair(s_feat, t_feat, ns, nt, eps=1e-12):
    xs = s_feat.flatten(1, -2).float()
    xt = t_feat.flatten(1, -2).float().detach()
    gram = torch.cat([xs.mT @ xs, xt.mT @ xt], 0)
    gram = gram + 1e-6 * torch.eye(xs.shape[2], device=gram.device, dtype=gram.dtype)
    evals, evecs = torch.linalg.eigh(gram)
    evals, evecs = evals.flip(-1), evecs.flip(-1)
    B = xs.shape[0]
    s_v = F.normalize(evecs[:B, :, :ns], dim=1)
    t_v = F.normalize(evecs[B:, :, :nt], dim=1)
    t_s = F.normalize(evals[B:, :nt].clamp_min(eps).sqrt(), dim=1)
    return s_v, t_v, t_s


def kd_svd_loss(
    student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...],
    stage_percentages: tuple[float, ...] | list[float] = (0.0, 0.1, 0.3, 0.6),
    rank: int = 3,
    rank_extras: int = 3,
    rbf_sigma: float = 8.0,
    gamma: float = 32.0,
) -> torch.Tensor:
    """Code implementation of the knowledge distillation using svd method.

    The original paper is Self-supervised Knowledge Distillation Using Singular Value Decomposition
    and can be found at https://arxiv.org/pdf/1807.06819

    Code implementation was originally from https://github.com/sseung0703/SSKD_SVD

    Args:
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): List of intermediate activations for the student model.
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...]): List of intermediate activations for the teacher model.
        stage_percentages (tuple[float] | list[float], optional): Tuple of floats representing each stage's contribution. Defaults to (0.0, 0.1, 0.3, 0.6).
        rank (int, optional): The top ranks of the SVD to choose. Defaults to 3.
        rank_extras (int, optional): The additional top ranks kept from the student feature. Defaults to 3.
        rbf_sigma (float, optional): The number applied to the rbf to smoothen it. Defaults to 8.0.
        gamma (float, optional): Multiplies the final loss value by gamma. Defaults to 32.0.

    Returns:
        torch.Tensor: The distillation loss.
    """
    loss = 0.0
    i = 0
    prev_student = None
    prev_teacher = None

    for teacher_stage, student_stage, stage_percentage in zip(
        teacher_stages, student_stages, stage_percentages
    ):
        if teacher_stage.shape[2:] != student_stage.shape[2:]:
            student_stage = F.interpolate(
                student_stage, size=teacher_stage.shape[2:], mode="nearest"
            )

        # # SVD method is here, it is way too slow for normal use
        # _, _, student_v = _get_svd(student_stage, rank + rank_extras)
        # _, teacher_s, teacher_v = _get_svd(teacher_stage, rank)
        # student_dfv, teacher_dfv = _align_rsv(student_v, teacher_v)
        # teacher_scaling = teacher_s.unsqueeze(1)
        # student_dfv, teacher_dfv = (
        #     student_dfv * teacher_scaling,
        #     teacher_dfv * teacher_scaling,
        # )
        #
        student_v, teacher_v, teacher_s = _top_rsv_pair(
            student_stage, teacher_stage, rank + rank_extras, rank
        )
        student_v = torch.nan_to_num(student_v, nan=0.0, posinf=0.0, neginf=0.0)

        student_dfv, teacher_dfv = _align_rsv(student_v, teacher_v)
        teacher_scaling = teacher_s.unsqueeze(1)
        student_dfv = student_dfv * teacher_scaling
        teacher_dfv = teacher_dfv * teacher_scaling

        if prev_student is not None:
            student_rbf = torch.exp(
                -(student_dfv.unsqueeze(2) - prev_student.unsqueeze(1)).square()
                / rbf_sigma
            )
            teacher_rbf = torch.exp(
                -(teacher_dfv.unsqueeze(2) - prev_teacher.unsqueeze(1)).square()  # type: ignore
                / rbf_sigma
            )

            l2 = (student_rbf - teacher_rbf.detach()).square()
            # Mask possible nans so to not backward them
            loss += (
                torch.nan_to_num(l2, nan=0.0, posinf=0.0, neginf=0.0).sum()
                * stage_percentage
            )
        i = i + 1
        prev_student = student_dfv  # type: ignore
        prev_teacher = teacher_dfv  # type: ignore
    return loss * gamma  # type: ignore
