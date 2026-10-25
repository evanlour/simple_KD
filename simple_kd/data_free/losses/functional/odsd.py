import torch
import torch.nn.functional as F


def instance_contrastive_loss(
    basic_input: torch.Tensor,
    augmented_input: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """Computes the instance contrastive input"""

    normalized_basic = F.normalize(basic_input, p=2, dim=1)
    normalized_aug = F.normalize(augmented_input, p=2, dim=1)
    n = normalized_basic.size(0)

    z = torch.cat([normalized_basic, normalized_aug], dim=0)

    sim_matrix = torch.mm(normalized_basic, z.t()) / tau  # All pairwise similarities
    positives = sim_matrix[
        torch.arange(n), torch.arange(n) + n
    ]  # The upper equation part of lc1

    # Mask out self-contrast (k == i in denominator)
    mask = torch.ones_like(sim_matrix, dtype=torch.bool)
    mask[torch.arange(n), torch.arange(n)] = False

    exp_sim = torch.exp(sim_matrix) * mask
    denom = exp_sim.sum(dim=1)

    l_c1 = -torch.log(torch.exp(positives) / denom).mean()
    return l_c1


def teacher_student_contrastive_loss(
    student_all: torch.Tensor,
    teacher_all: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """Computes the teacher student contrastive loss.

    Args:
        student_all: Concatenated [student_weak, student_strong] (shape [2N, C])
        teacher_all: Concatenated [teacher_weak, teacher_strong] (shape [2N, C])
    """
    z_s = F.normalize(student_all, p=2, dim=1)
    z_t = F.normalize(teacher_all, p=2, dim=1)
    two_n = z_s.size(0)

    sim_matrix = torch.mm(z_t, z_s.t()) / tau
    positives = torch.diagonal(sim_matrix)

    # Mask out self-comparison (k == i) in denominator
    mask = ~torch.eye(two_n, dtype=torch.bool, device=z_s.device)
    exp_sim = torch.exp(sim_matrix) * mask
    denom = exp_sim.sum(dim=1)

    l_c2 = -torch.log(torch.exp(positives) / denom).mean()
    return l_c2


def spatial_mapping_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Computes the SVD-based Spatial Mapping loss between student and teacher representations.

    Uses Singular Value Decomposition (SVD) on regularized minibatches, normalizes the resulting
    spatial representations (U * S), and computes a Huber loss between student and teacher.
    Computations are performed in float32 for numerical stability under mixed precision.

    Args:
        student_output (torch.Tensor): Logits/features from the student model (shape: [B, C]).
        teacher_output (torch.Tensor): Logits/features from the teacher model (shape: [B, C]).
        huber_delta (float, optional): Delta parameter for HuberLoss. Defaults to 1.0.

    Returns:
        torch.Tensor: Scalar spatial mapping loss.
    """
    teacher_fp32 = teacher_output.float()
    student_fp32 = student_output.float()

    t_b, t_c = teacher_fp32.shape
    s_b, s_c = student_fp32.shape

    teacher_minibatch = (
        teacher_fp32 + torch.eye(t_b, m=t_c, device=teacher_fp32.device) * 1e-6
    )
    student_minibatch = (
        student_fp32 + torch.eye(s_b, m=s_c, device=student_fp32.device) * 1e-6
    )

    teacher_u, teacher_s, _ = torch.linalg.svd(teacher_minibatch, full_matrices=False)
    student_u, student_s, _ = torch.linalg.svd(student_minibatch, full_matrices=False)

    teacher_spatial = F.normalize(teacher_u * teacher_s, p=2, dim=1)
    student_spatial = F.normalize(student_u * student_s, p=2, dim=1)

    loss = F.huber_loss(student_spatial, teacher_spatial, delta=huber_delta)
    return loss.to(student_output.dtype)


def odsd_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    student_basic_output: torch.Tensor,
    teacher_basic_output: torch.Tensor,
    kd_temperature: float = 4.0,
    kd_modifier: float = 1.0,
    spatial_modifier: float = 4.0,
    relational_modifier: float = 4.0,
    huber_delta: float = 1.0,
    tau: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes the combined Out-of-Distribution Sample Distillation (ODSD) loss.

    The method is proposed in: Sampling to Distill: Knowledge Transfer from Open-World Data (https://arxiv.org/pdf/2307.16601v2)

    The total distillation loss is comprised of three components: The KD loss, the spatial loss and the
    denoising contrastive relational loss. In this implementation, the barlow twins loss is implemented as it performed better at a sample of tests.

    Args:
        student_output (torch.Tensor): Student predictions on strongly augmented inputs.
        teacher_output (torch.Tensor): Teacher predictions on strongly augmented inputs.
        student_basic_output (torch.Tensor): Student predictions on weakly/basic augmented inputs.
        teacher_basic_output (torch.Tensor): Teacher predictions on weakly/basic augmented inputs.
        kd_temperature (float, optional): Softmax temperature for KD. Defaults to 4.0.
        kd_modifier (float, optional): Multiplier for KD loss. Defaults to 1.0.
        spatial_modifier (float, optional): Multiplier for spatial mapping loss. Defaults to 4.0.
        relational_modifier (float, optional): Multiplier for Barlow Twins relational loss. Defaults to 4.0.
        huber_delta (float, optional): Delta parameter for Huber loss in spatial mapping. Defaults to 1.0.
        tau (float, optional): Constrastive relational loss temperature. Defaults to 0.5.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - total_loss: Weighted sum of all three loss terms.
            - loss_components: Dictionary with individual unweighted loss values ("kd_loss", "spatial_loss", "relational_loss").
    """
    # KD Loss
    log_pred_student = F.log_softmax(student_output / kd_temperature, dim=1)
    pred_teacher = F.softmax(teacher_output / kd_temperature, dim=1)
    lkd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction="batchmean")
        * (kd_temperature**2)
        * kd_modifier
    )

    # Spatial Mapping Loss
    spatial_loss = (
        spatial_mapping_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            huber_delta=huber_delta,
        )
        * spatial_modifier
    )

    # Contrastive Relational Loss
    combined_student = torch.cat([student_basic_output, student_output], dim=0)
    combined_teacher = torch.cat([teacher_basic_output, teacher_output], dim=0)

    contrastive_loss1 = instance_contrastive_loss(
        student_basic_output,
        student_output,
        tau=tau,
    )
    contrastive_loss2 = teacher_student_contrastive_loss(
        combined_student,
        combined_teacher,
        tau=tau,
    )
    relational_loss = (contrastive_loss1 + contrastive_loss2) * relational_modifier

    total_loss = lkd_loss + spatial_loss + relational_loss

    loss_dict = {
        "kd_loss": lkd_loss,
        "spatial_loss": spatial_loss,
        "relational_loss": relational_loss,
    }

    return total_loss, loss_dict
