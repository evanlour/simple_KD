import csv
import time
from pathlib import Path

import torch
from accelerate import Accelerator

from simple_kd.training import evaluate_model
from simple_kd.utils import SimpleGradClip

from .common import images_from_generator, train_generator, train_model_with_generator
from .losses import ZSKTLoss


def zskt_train(
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
    generator_steps: int = 1,
    model_steps: int = 10,
    print_frequency: int = 10,
    temperature: float = 4.0,
    model_multipler: float = 1.0,
    generator_multiplier: float = 1.0,
    model_at_gamma: float = 0.0,
    stage_percentages: list[float] | tuple[float, ...] = (0.1, 0.1, 0.3, 0.6),
    validation_frequency: int = 50,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    generator_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...] = (),
    teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...] = (),
    clipper: SimpleGradClip | None = None,
    validation_metrics: dict | None = None,
    save_location: Path | None = None,
    save_prefix: str = "",
    save_student_model: bool = True,
    save_generator: bool = False,
    save_history: bool = True,
) -> dict:
    """Trains a student model and generator using Zero-Shot Knowledge Transfer (ZSKT).

    ZSKT optimizes a generator and a student model in an adversarial game:
    - The generator is optimized to produce samples on which the student and teacher disagree,
      maximizing the KL divergence between their predictions (sign = -1). Optionally, the
      attention transfer discrepancy can also be maximized via generator_at_gamma.
    - The student is trained to minimize the KL divergence to the teacher, plus an
      Attention Transfer (AT) term on intermediate stages (sign = +1).
    Each epoch runs generator_steps generator updates followed by model_steps student updates.

    The original paper is "Zero-shot Knowledge Transfer via Adversarial Belief Matching"
    (Micaelli & Storkey, NeurIPS 2019, https://arxiv.org/abs/1905.09768). The code is available
    here https://github.com/polo5/ZeroShotKnowledgeTransfer

    Args:
        accelerator (Accelerator): The HuggingFace Accelerator instance.
        student_model (torch.nn.Module): The student model to be trained.
        teacher_model (torch.nn.Module): The pre-trained frozen teacher model.
        generator (torch.nn.Module): Generative network mapping latent noise vectors to synthetic images.
        augmentations (torch.nn.Module): On-device augmentations and normalizations applied to generated images.
        student_optimizer (torch.optim.Optimizer): Optimizer for student model weights.
        generator_optimizer (torch.optim.Optimizer): Optimizer for generator network weights.
        epochs (int): Total number of training epochs (ZSKT iterations).
        batch_size (int): Batch size of synthetic images per step.
        generator_latent_size (int): Dimensionality of the input latent noise vector z.
        generator_steps (int, optional): Number of generator updates per epoch (n_G in the paper). Defaults to 1.
        model_steps (int, optional): Number of student updates per epoch (n_S in the paper). Defaults to 10.
        print_frequency (int, optional): Frequency (in epochs) to print progress to console. Defaults to 10.
        temperature (float, optional): Distillation temperature for the KL term. Defaults to 4.0.
        model_multipler (float, optional): Multiplier/gamma for the student KD loss. Defaults to 1.0.
        generator_multiplier (float, optional): Multiplier/gamma for the generator KD loss. Defaults to 1.0.
        model_at_gamma (float, optional): Multiplier/gamma for the student AT loss (beta in the paper). Defaults to 128.0.
        stage_percentages (list[float] | tuple[float, ...], optional): Per-stage weights for the AT loss.
            Must match the number of stages. Defaults to (0.1, 0.1, 0.3, 0.6).
        validation_frequency (int, optional): Frequency (in epochs) to evaluate on validation metrics. Defaults to 50.
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler for student. Defaults to None.
        generator_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler for generator. Defaults to None.
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...], optional): Intermediate student activation hooks. Defaults to [].
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...], optional): Intermediate teacher activation hooks. Defaults to [].
        clipper (SimpleGradClip, optional): Gradient clipping utility. Defaults to None.
        validation_metrics (dict, optional): Dictionary with validation images, labels, augs, and classes for periodic evaluation. Defaults to None.
        save_location (Path, optional): Directory where checkpoints and history CSV will be stored. Defaults to None.
        save_prefix (str, optional): Prefix for saved artifact filenames. Defaults to "".
        save_student_model (bool, optional): Whether to save student model checkpoint. Defaults to True.
        save_generator (bool, optional): Whether to save generator checkpoint. Defaults to False.
        save_history (bool, optional): Whether to save metrics history to CSV. Defaults to True.

    Returns:
        dict: Training history containing per-epoch metric lists:
            - "student_loss": Mean student ZSKT loss per epoch.
            - "generator_loss": Mean generator adversarial loss per epoch.
            - "student_lr": Student optimizer learning rate per epoch.
            - "generator_lr": Generator optimizer learning rate per epoch.
            - "gen_conf": Mean confidence (max probability) of teacher on generator images.
            - "gen_entropy": Mean normalized class diversity entropy across batches.
            - "gen_spread": Mean spatial pixel standard deviation of generated images.
            - "epoch_time": Total elapsed time in seconds per epoch.
    """
    criterion = ZSKTLoss(
        T=temperature,
        kd_gamma=model_multipler,
        at_gamma=model_at_gamma,
        stage_percentages=stage_percentages,
        sign=1.0,
    ).to(accelerator.device)
    generator_criterion = ZSKTLoss(
        T=temperature,
        kd_gamma=generator_multiplier,
        at_gamma=0.0,
        stage_percentages=stage_percentages,
        sign=-1.0,
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
        total_gen_loss = torch.tensor(0.0, device=accelerator.device)
        gen_conf = torch.tensor(0.0, device=accelerator.device)
        gen_entropy = torch.tensor(0.0, device=accelerator.device)
        gen_spread = torch.tensor(0.0, device=accelerator.device)

        model_train_total = torch.tensor(0, device=accelerator.device)
        generator_train_total = torch.tensor(0, device=accelerator.device)
        start_time = time.perf_counter()

        # Generator phase
        generator.train()
        student_model.train()
        for _ in range(generator_steps):
            gen_imgs = images_from_generator(
                generator,
                augmentations,
                accelerator.device,
                batch_size,
                generator_latent_size,
                requires_grad=True,
            )
            gen_loss = train_generator(
                accelerator,
                gen_imgs,
                student_model,
                teacher_model,
                generator,
                generator_criterion,
                generator_optimizer,
                student_stages=student_stages,
                teacher_stages=teacher_stages,
                clipper=clipper,
            )
            total_gen_loss += gen_loss.detach() * batch_size
            generator_train_total += batch_size
            # print(gen_loss)

        # Student phase: minimize KD + AT discrepancy
        generator.train()
        student_model.train()
        for _ in range(model_steps):
            gen_imgs = images_from_generator(
                generator,
                augmentations,
                accelerator.device,
                batch_size,
                generator_latent_size,
                requires_grad=True,
            )
            train_loss, gen_statistics = train_model_with_generator(
                accelerator,
                gen_imgs,
                student_model,
                teacher_model,
                criterion,
                student_optimizer,
                student_stages=student_stages,
                teacher_stages=teacher_stages,
                clipper=clipper,
            )
            # print(train_loss)
            total_train_loss += train_loss.detach() * batch_size  # type: ignore
            model_train_total += batch_size
            gen_conf += gen_statistics["gen_conf"] * batch_size
            gen_entropy += gen_statistics["gen_entropy"] * batch_size
            gen_spread += gen_statistics["gen_spread"] * batch_size

        model_epoch_samples = accelerator.reduce(
            model_train_total, reduction="sum"
        ).item()  # type: ignore
        generator_epoch_samples = accelerator.reduce(
            generator_train_total, reduction="sum"
        ).item()  # type: ignore
        model_epoch_loss = (
            accelerator.reduce(total_train_loss, reduction="sum") / model_epoch_samples
        )
        gen_epoch_loss = (
            accelerator.reduce(total_gen_loss, reduction="sum")
            / generator_epoch_samples
        )
        gen_conf = (
            accelerator.reduce(gen_conf, reduction="sum").item() / model_epoch_samples  # type: ignore
        )
        gen_entropy = (
            accelerator.reduce(gen_entropy, reduction="sum").item()  # type: ignore
            / model_epoch_samples
        )
        gen_spread = (
            accelerator.reduce(gen_spread, reduction="sum").item() / model_epoch_samples  # type: ignore
        )

        if scheduler is not None:
            scheduler.step()
        if generator_scheduler is not None:
            generator_scheduler.step()

        if accelerator.is_main_process:
            epoch_time = time.perf_counter() - start_time
            history["student_loss"].append(model_epoch_loss.item())
            history["student_lr"].append(student_optimizer.param_groups[0]["lr"])
            history["generator_loss"].append(gen_epoch_loss.item())
            history["generator_lr"].append(generator_optimizer.param_groups[0]["lr"])
            history["gen_conf"].append(gen_conf)
            history["gen_entropy"].append(gen_entropy)
            history["gen_spread"].append(gen_spread)
            history["epoch_time"].append(epoch_time)

            if epoch % print_frequency == 0:
                temp_message = f"Epoch {epoch + 1}/{epochs} Student Train Loss: {model_epoch_loss:.4f}, Student lr: {student_optimizer.param_groups[0]['lr']:.6f},"
                temp_message += f" Generator confidence: {gen_conf:.4f}, Generator Entropy: {gen_entropy:.4f}, Generator spread: {gen_spread:.4f},"
                temp_message += f" Generator Train Loss: {gen_epoch_loss:.4f}, Generator lr: {generator_optimizer.param_groups[0]['lr']:.6f}, Training time: {epoch_time:.2f}"
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
