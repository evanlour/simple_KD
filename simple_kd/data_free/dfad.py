import csv
import time
from pathlib import Path

import torch
from accelerate import Accelerator

from simple_kd.training import evaluate_model
from simple_kd.utils import SimpleGradClip

from .common import images_from_generator, train_generator, train_model_with_generator
from .losses import DFADLoss


def dfad_train(
    accelerator: Accelerator,
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    generator: torch.nn.Module,
    augmentations: torch.nn.Module,
    student_optimizer: torch.optim.Optimizer,
    generator_optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    generator_latent_size: int,
    steps_per_epoch: int = 50,
    print_frequency: int = 10,
    model_multipler: float = 1.0,
    generator_multiplier: float = 1.0,
    use_log1p: bool = False,
    validation_frequency: int = 50,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    generator_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    clipper: SimpleGradClip | None = None,
    validation_metrics: dict | None = None,
    save_location: Path | None = None,
    save_prefix: str = "",
    save_student_model: bool = True,
    save_generator: bool = False,
    save_history: bool = True,
) -> dict:
    """Trains a student model and generator using Data-Free Adversarial Distillation (DFAD).

    DFAD optimizes a generator and a student model in an adversarial zero-sum game:
    - The generator is optimized to produce difficult synthetic samples that maximize
      the discrepancy between student and teacher predictions (sign = -1).
    - The student is trained to minimize this discrepancy (sign = +1), effectively matching
      the teacher across the synthesized data manifold without needing original training images.

    The original paper is "Data-Free Adversarial Distillation" (Fang et al., CVPR 2020), https://arxiv.org/abs/1912.11006.
    The original code can be found at https://github.com/VainF/Data-Free-Adversarial-Distillation.

    Args:
        accelerator (Accelerator): The HuggingFace Accelerator instance.
        student_model (torch.nn.Module): The student model to be trained.
        teacher_model (torch.nn.Module): The pre-trained frozen teacher model.
        generator (torch.nn.Module): Generative network mapping latent noise vectors to synthetic images.
        augmentations (torch.nn.Module): On-device augmentations and normalizations applied to generated images.
        student_optimizer (torch.optim.Optimizer): Optimizer for student model weights.
        generator_optimizer (torch.optim.Optimizer): Optimizer for generator network weights.
        epochs (int): Total number of training epochs.
        batch_size (int): Batch size of synthetic images per step.
        generator_latent_size (int): Dimensionality of the input latent noise vector z.
        steps_per_epoch (int, optional): Number of student training steps per epoch. Defaults to 50.
        print_frequency (int, optional): Frequency (in epochs) to print progress to console. Defaults to 10.
        model_multipler (float, optional): Multiplier/gamma for the student DFAD loss. Defaults to 1.0.
        generator_multiplier (float, optional): Multiplier/gamma for the generator DFAD loss. Defaults to 1.0.
        use_log1p (bool, optional): Whether to wrap the L1 loss in torch.log1p (ADA protocol). Defaults to False.
        validation_frequency (int, optional): Frequency (in epochs) to evaluate on validation metrics. Defaults to 50.
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler for student. Defaults to None.
        generator_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler for generator. Defaults to None.
        clipper (SimpleGradClip, optional): Gradient clipping utility. Defaults to None.
        validation_metrics (dict, optional): Dictionary with validation images, labels, augs, and classes for periodic evaluation. Defaults to None.
        save_location (Path, optional): Directory where checkpoints and history CSV will be stored. Defaults to None.
        save_prefix (str, optional): Prefix for saved artifact filenames. Defaults to "".
        save_student_model (bool, optional): Whether to save student model checkpoint. Defaults to True.
        save_generator (bool, optional): Whether to save generator checkpoint. Defaults to False.
        save_history (bool, optional): Whether to save metrics history to CSV. Defaults to True.

    Returns:
        dict: Training history containing per-epoch metric lists:
            - "student_loss": Mean student discrepancy loss per epoch.
            - "generator_loss": Mean generator adversarial loss per epoch.
            - "student_lr": Student optimizer learning rate per epoch.
            - "generator_lr": Generator optimizer learning rate per epoch.
            - "gen_conf": Mean confidence (max probability) of teacher on generator images.
            - "gen_entropy": Mean normalized class diversity entropy across batches.
            - "gen_spread": Mean spatial pixel standard deviation of generated images.
            - "epoch_time": Total elapsed time in seconds per epoch.
    """
    criterion = DFADLoss(sign=1.0 * model_multipler, use_log1p=use_log1p).to(
        accelerator.device
    )
    generator_criterion = DFADLoss(
        sign=-1.0 * generator_multiplier, use_log1p=use_log1p
    ).to(accelerator.device)
    history = {
        "student_loss": [],
        "generator_loss": [],
        "student_lr": [],
        "generator_lr": [],
        "gen_conf": [],
        "gen_entropy": [],
        "gen_spread": [],
        "epoch_time": [],
    }
    teacher_model.eval()
    for epoch in range(epochs):
        total_train_loss = torch.tensor(0.0, device=accelerator.device)
        gen_conf = torch.tensor(0.0, device=accelerator.device)
        gen_entropy = torch.tensor(0.0, device=accelerator.device)
        gen_spread = torch.tensor(0.0, device=accelerator.device)
        generator.train()
        student_model.train()

        train_total = torch.tensor(0, device=accelerator.device)
        start_time = time.perf_counter()
        for _ in range(steps_per_epoch):
            gen_imgs = images_from_generator(
                generator,
                augmentations,
                accelerator.device,
                batch_size,
                generator_latent_size,
            ).detach()
            train_loss, gen_statistics = train_model_with_generator(
                accelerator,
                gen_imgs,
                student_model,
                teacher_model,
                criterion,
                student_optimizer,
                student_stages=None,
                teacher_stages=None,
                clipper=clipper,
            )
            total_train_loss += train_loss.detach() * batch_size  # type: ignore
            train_total += batch_size
            gen_conf += gen_statistics["gen_conf"] * batch_size
            gen_entropy += gen_statistics["gen_entropy"] * batch_size
            gen_spread += gen_statistics["gen_spread"] * batch_size

        epoch_samples = accelerator.reduce(train_total, reduction="sum").item()  # type: ignore
        model_epoch_loss = accelerator.reduce(total_train_loss, reduction="sum")
        model_epoch_loss = model_epoch_loss / epoch_samples
        gen_conf = accelerator.reduce(gen_conf, reduction="sum").item() / epoch_samples  # type: ignore
        gen_entropy = (
            accelerator.reduce(gen_entropy, reduction="sum").item() / epoch_samples  # type: ignore
        )
        gen_spread = (
            accelerator.reduce(gen_spread, reduction="sum").item() / epoch_samples  # type: ignore
        )

        gen_imgs = images_from_generator(
            generator,
            augmentations,
            accelerator.device,
            batch_size,
            generator_latent_size,
            requires_grad=True,
        ).detach()
        gen_loss = train_generator(
            accelerator,
            gen_imgs,
            student_model,
            teacher_model,
            generator,
            generator_criterion,
            generator_optimizer,
            student_stages=None,
            teacher_stages=None,
            clipper=clipper,
        )

        if scheduler is not None:
            scheduler.step()
        if generator_scheduler is not None:
            generator_scheduler.step()

        if accelerator.is_main_process:
            epoch_time = time.perf_counter() - start_time
            history["student_loss"].append(model_epoch_loss.item())
            history["student_lr"].append(student_optimizer.param_groups[0]["lr"])
            history["generator_loss"].append(gen_loss.item())
            history["generator_lr"].append(generator_optimizer.param_groups[0]["lr"])
            history["gen_conf"].append(gen_conf)
            history["gen_entropy"].append(gen_entropy)
            history["gen_spread"].append(gen_spread)
            history["epoch_time"].append(epoch_time)

            if epoch % print_frequency == 0:
                temp_message = f"Epoch {epoch + 1}/{epochs} Student Train Loss: {model_epoch_loss:.4f}, Student lr: {student_optimizer.param_groups[0]['lr']:.6f},"
                temp_message += f" Generator confidence: {gen_conf:.4f}, Generator Entropy: {gen_entropy:.4f}, Generator spread: {gen_spread:.4f},"
                temp_message += f" Generator Train Loss: {gen_loss:.4f}, Generator lr: {generator_optimizer.param_groups[0]['lr']:.6f}, Training time: {epoch_time:.2f}"
                print(temp_message)

            if epoch % validation_frequency == 0 and validation_metrics is not None:
                _ = evaluate_model(
                    model=student_model,
                    images=validation_metrics["validation_images"],
                    labels=validation_metrics["validation_labels"],
                    batch_size=batch_size,
                    device=accelerator.device,
                    augs=validation_metrics["validation_augs"],
                    criterion=torch.nn.CrossEntropyLoss(),
                    classes=validation_metrics["classes"],
                    save_folder=None,
                )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and save_location is not None:
        if save_student_model:
            student_model_name = (
                f"{save_prefix}_student_model.pth"
                if save_prefix
                else "student_model.pth"
            )
            student_model_savepath = save_location / student_model_name
            student_model_savepath.parent.mkdir(parents=True, exist_ok=True)
            unwrapped_student_model = accelerator.unwrap_model(
                student_model, keep_torch_compile=False
            )
            torch.save(unwrapped_student_model.state_dict(), student_model_savepath)

        if save_generator:
            generator_name = (
                f"{save_prefix}_generator.pth" if save_prefix else "generator.pth"
            )
            generator_savepath = save_location / generator_name
            generator_savepath.parent.mkdir(parents=True, exist_ok=True)
            unwrapped_generator = accelerator.unwrap_model(
                generator, keep_torch_compile=False
            )
            torch.save(unwrapped_generator.state_dict(), generator_savepath)

        if save_history:
            history_name = (
                f"{save_prefix}_history.csv" if save_prefix else "history.csv"
            )
            history_savepath = save_location / history_name
            history_savepath.parent.mkdir(parents=True, exist_ok=True)
            data = {
                key: vals + [vals[-1]] * (epochs - len(vals))
                for key, vals in history.items()
            }
            with open(history_savepath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data.keys())
                writer.writerows(zip(*data.values()))

    accelerator.free_memory()
    return history
