import inspect

import torch
from accelerate import Accelerator

from simple_kd.utils import SimpleGradClip


def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    """Passes only kwargs that fn's signature actually accepts, ignoring None.

    Args:
        fn (callable): A callable or PyTorch Module whose signature will be inspected.
        kwargs (dict): Dictionary of candidate keyword arguments.

    Returns:
        dict: Filtered dictionary containing only arguments accepted by fn.
    """
    target = getattr(fn, "forward", fn)
    sig = inspect.signature(target)
    valid_names = set(sig.parameters.keys())
    return {
        k: v for k, v in kwargs.items()
        if k in valid_names and v is not None
    }

def compute_generator_metrics(images: torch.Tensor, teacher_logits: torch.Tensor) -> dict:
    """Computes quality, diversity, and spread metrics directly from step tensors.

    Args:
        images (torch.Tensor): Batch of synthetic images produced by the generator.
        teacher_logits (torch.Tensor): Teacher model logits evaluated on the synthetic images.

    Returns:
        dict: A dictionary containing:
            - "gen_conf" (float): Mean maximum softmax probability across the batch, indicating
              semantic fidelity / categorical sharpness (Ideal: > 0.70).
            - "gen_entropy" (float): Batch-level Shannon entropy normalized by log(num_classes),
              measuring categorical diversity across classes (Ideal: 0.70 - 0.95).
            - "gen_spread" (float): Spatial pixel standard deviation, ruling out flat gray or
              checkerboard generator collapse (Ideal: > 0.15).
    """
    with torch.no_grad():
        probs = torch.softmax(teacher_logits, dim=-1)

        conf = probs.max(dim=-1).values.mean().item() # Quality / Semantic Fidelity (Ideal: > 0.70)

        # Batch-level Class Diversity (Ideal: > 0.70 - 0.90)
        mean_dist = probs.mean(dim=0)
        num_classes = probs.size(-1)
        entropy = -(mean_dist * torch.log(mean_dist + 1e-8)).sum().item()
        norm_entropy = entropy / torch.log(torch.tensor(float(num_classes))).item()

        pixel_spread = images.std(dim=0).mean().item() # Spatial/Pixel Diversity (Rules out flat gray/checkerboard collapse; Ideal: > 0.15)

    return {
        "gen_conf": conf,
        "gen_entropy": norm_entropy,
        "gen_spread": pixel_spread,
    }

def images_from_generator(generator: torch.nn.Module,
                          augmentations: torch.nn.Module,
                          device: torch.device,
                          batch_size: int,
                          generator_latent_size: int,
                          requires_grad: bool=False) -> torch.Tensor:
    """Generates a batch of synthetic images from random Gaussian latent vectors.

    Args:
        generator (torch.nn.Module): The generative model mapping latent noise to images.
        augmentations (torch.nn.Module): On-device augmentations/normalizations applied to images.
        device (torch.device): Device where latent vectors and computations reside.
        batch_size (int): Number of synthetic images to generate.
        generator_latent_size (int): Dimensionality of the input latent noise vector.
        requires_grad (bool, optional): If True, retains gradients for generator optimization.
            If False, executes under torch.no_grad(). Defaults to False.

    Returns:
        torch.Tensor: Batch of augmented synthetic images ready for model consumption.
    """
    z = torch.randn(batch_size, generator_latent_size, device=device)
    if requires_grad:
        generator.train()
        gen_imgs = generator(z)
        gen_imgs = (gen_imgs + 1.0) / 2.0
        gen_imgs = augmentations(gen_imgs)
    else:
        generator.eval()
        with torch.no_grad():
            gen_imgs = generator(z)
            gen_imgs = (gen_imgs + 1.0) / 2.0
            gen_imgs = augmentations(gen_imgs)
    return gen_imgs

def train_model_with_generator(accelerator: Accelerator,
                                images: torch.Tensor,
                                student_model: torch.nn.Module,
                                teacher_model: torch.nn.Module,
                                generator: torch.nn.Module,
                                criterion: torch.nn.Module,
                                student_optimizer: torch.optim.Optimizer,
                                student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...] = (),
                                teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...] = (),
                                clipper: SimpleGradClip | None = None,) -> tuple[torch.Tensor, dict]:
    """Executes a single optimization step for the student model using generator images.

    Evaluates the student on synthetic images, gathers teacher outputs under no_grad,
    computes generator quality metrics, evaluates distillation criterion, and updates student weights.

    Args:
        accelerator (Accelerator): The HuggingFace Accelerator instance.
        images (torch.Tensor): Synthetic images batch produced by the generator.
        student_model (torch.nn.Module): Student model being trained.
        teacher_model (torch.nn.Module): Pre-trained teacher model providing distillation targets.
        generator (torch.nn.Module): Generator model (placed in eval mode).
        criterion (torch.nn.Module): Student distillation loss criterion (e.g. DFADLoss, KDLoss).
        student_optimizer (torch.optim.Optimizer): Optimizer for student model parameters.
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...], optional): Student intermediate stage activations. Defaults to [].
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...], optional): Teacher intermediate stage activations. Defaults to [].
        clipper (SimpleGradClip, optional): Optional gradient clipping utility. Defaults to None.

    Returns:
        tuple[torch.Tensor, dict]: Tuple containing:
            - train_loss (torch.Tensor): The scalar loss tensor computed by criterion.
            - generator_statistics (dict): Dictionary of generator metrics (conf, entropy, spread).
    """
    train_loss = torch.tensor(0.0, device=accelerator.device)
    generator.eval()
    student_model.train()
    with accelerator.accumulate(student_model):
        student_output = student_model(images)
        with torch.no_grad():
            teacher_output = teacher_model(images)
            generator_statistics = compute_generator_metrics(images, teacher_output)
        available_args = {
            "student_output": student_output,
            "teacher_output": teacher_output.detach() if teacher_output is not None else None,
            "student_stages": student_stages,
            "teacher_stages": [v.detach() for v in teacher_stages] if teacher_stages is not None else None,
        }
        filtered_args = _filter_kwargs_for_callable(criterion, available_args)
        train_loss = criterion(**filtered_args)
        accelerator.backward(train_loss)

        if clipper is not None:
            accelerator.unscale_gradients(student_optimizer)
            clipper(student_model.parameters())
        student_optimizer.step()
        student_optimizer.zero_grad(set_to_none=True)
    return train_loss, generator_statistics


def train_generator(accelerator: Accelerator,
                                images: torch.Tensor,
                                student_model: torch.nn.Module,
                                teacher_model: torch.nn.Module,
                                generator: torch.nn.Module,
                                criterion: torch.nn.Module,
                                generator_optimizer: torch.optim.Optimizer,
                                student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...] = (),
                                teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...] = (),
                                clipper: SimpleGradClip | None = None,) -> torch.Tensor:
    """Executes a single optimization step for the generator.

    Feeds synthetic images through student and teacher models, calculates the adversarial
    discrepancy loss, and backpropagates gradients into the generator.

    Args:
        accelerator (Accelerator): The HuggingFace Accelerator instance.
        images (torch.Tensor): Synthetic images with requires_grad=True connected to the generator.
        student_model (torch.nn.Module): Student model (placed in eval mode).
        teacher_model (torch.nn.Module): Pre-trained teacher model.
        generator (torch.nn.Module): Generator model being optimized.
        criterion (torch.nn.Module): Generator loss criterion (e.g. inverted DFADLoss).
        generator_optimizer (torch.optim.Optimizer): Optimizer for generator parameters.
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...], optional): Student intermediate stage activations. Defaults to [].
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...], optional): Teacher intermediate stage activations. Defaults to [].
        clipper (SimpleGradClip, optional): Optional gradient clipping utility. Defaults to None.

    Returns:
        torch.Tensor: The scalar generator loss tensor.
    """
    train_loss = torch.tensor(0.0, device=accelerator.device)
    generator.train()
    student_model.eval()
    with accelerator.accumulate(generator):
        student_output = student_model(images)
        with torch.no_grad():
            teacher_output = teacher_model(images)
        available_args = {
            "student_output": student_output,
            "teacher_output": teacher_output.detach() if teacher_output is not None else None,
            "student_stages": student_stages,
            "teacher_stages": [v.detach() for v in teacher_stages] if teacher_stages is not None else None,
        }
        filtered_args = _filter_kwargs_for_callable(criterion, available_args)
        train_loss = criterion(**filtered_args)
        accelerator.backward(train_loss)

        if clipper is not None:
            accelerator.unscale_gradients(generator_optimizer)
            clipper(generator.parameters())
        generator_optimizer.step()
        generator_optimizer.zero_grad(set_to_none=True)

    return train_loss
