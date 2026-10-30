import torch
import torch.nn.functional as F
from torch.autograd import grad


def _get_normalized_grad(
    target: torch.Tensor,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    create_graph: bool = False,
):
    target_out = torch.gather(target, 1, labels.view(-1, 1))
    target_grad = grad(
        outputs=target_out,
        inputs=imgs,
        grad_outputs=torch.ones_like(target_out),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    target_grad = F.normalize(target_grad.reshape(target_grad.size(0), -1), p=2, dim=1)
    return target_grad


def sobolev_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 256.0,
) -> torch.Tensor:
    """Wrapper module for Sobolev loss.

    The original paper Sobolev Training for Neural Networks can be found at https://arxiv.org/pdf/1706.04859.

    Code implementation is mostly extracted from The mdistiller repository at https://github.com/megvii-research/mdistiller.

    Args:
        student_output (torch.Tensor): Output logits from the student model.
        teacher_output (torch.Tensor): Output logits from the teacher model.
        labels (torch.Tensor): Ground truth target labels.
        images (torch.Tensor): The input images.
        gamma (float, optional): The gamma the loss will be multiplied with. Defaults to 256.0.
    """
    student_grad = _get_normalized_grad(
        student_output, images, labels, create_graph=True
    )
    teacher_grad = _get_normalized_grad(teacher_output, images, labels)
    loss = F.mse_loss(student_grad, teacher_grad.detach())
    return loss * gamma
