import csv
import time
from pathlib import Path

import torch
from accelerate import Accelerator

from simple_kd.training import evaluate_model
from simple_kd.utils import SimpleGradClip

from .losses import ODSDLoss


def odsd_train(
    accelerator: Accelerator,
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    student_optimizer: torch.optim.Optimizer,
    epochs: int,
    augmentations: torch.nn.Module,
    basic_augmentations: torch.nn.Module,
    kd_temperature: float = 4.0,
    kd_loss_modifier: float = 1.0,
    spatial_loss_modifier: float = 4.0,
    relational_loss_modifier: float = 4.0,
    barlow_twins_lambda: float = 0.005,
    huber_delta: float = 1.0,
    print_frequency: int = 10,
    validation_frequency: int = 50,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...] = (),
    teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...] = (),
    clipper: SimpleGradClip | None = None,
    validation_metrics: dict | None = None,
    save_location: Path | None = None,
    save_prefix: str = "",
    save_student_model: bool = True,
    save_history: bool = True,
) -> dict:
    """Trains a student model The ODSD step 2 training pipeline.

    Distills a frozen pre-trained teacher into a student network on sampled proxy data using:
    - Temperature-scaled Knowledge Distillation (KD) loss.
    - SVD-based Spatial Mapping loss aligned via Huber loss.
    - Barlow Twins dual-view cross-correlation contrastive loss for relational representation alignment.

    The method is proposed in: Sampling to Distill: Knowledge Transfer from Open-World Data (https://arxiv.org/pdf/2307.16601v2)

    Args:
        accelerator (Accelerator): The HuggingFace Accelerator instance.
        student_model (torch.nn.Module): The student model to be trained.
        teacher_model (torch.nn.Module): The pre-trained frozen teacher model.
        train_loader (torch.utils.data.DataLoader): DataLoader supplying proxy training batches.
        student_optimizer (torch.optim.Optimizer): Optimizer for student model weights.
        epochs (int): Total number of distillation epochs.
        augmentations (torch.nn.Module | None, optional): On-device strong augmentations applied to images.
        basic_augmentations (torch.nn.Module | None, optional): On-device basic/weak augmentations applied to images.
        kd_temperature (float, optional): Softmax temperature for KD. Defaults to 4.0.
        kd_loss_modifier (float, optional): Multiplier for KD loss. Defaults to 1.0.
        spatial_loss_modifier (float, optional): Multiplier for spatial mapping loss. Defaults to 4.0.
        relational_loss_modifier (float, optional): Multiplier for Barlow Twins relational loss. Defaults to 4.0.
        barlow_twins_lambda (float, optional): Lambda parameter for off-diagonal Barlow Twins terms. Defaults to 0.005.
        huber_delta (float, optional): Delta parameter for Huber loss in spatial mapping. Defaults to 1.0.
        print_frequency (int, optional): Frequency (in epochs) to print progress to console. Defaults to 10.
        validation_frequency (int, optional): Frequency (in epochs) to evaluate on validation metrics. Defaults to 50.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None, optional): Learning rate scheduler for student. Defaults to None.
        student_stages (list[torch.Tensor] | tuple[torch.Tensor, ...], optional): Intermediate student activation hooks. Defaults to ().
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...], optional): Intermediate teacher activation hooks. Defaults to ().
        clipper (SimpleGradClip | None, optional): Optional gradient clipping utility. Defaults to None.
        validation_metrics (dict | None, optional): Dict with validation images, labels, augs, and classes for evaluation.
        save_location (Path | None, optional): Directory where checkpoints and history CSV will be stored. Defaults to None.
        save_prefix (str, optional): Prefix prepended to saved artifact filenames. Defaults to "".
        save_student_model (bool, optional): Whether to save student model checkpoint. Defaults to True.
        save_history (bool, optional): Whether to save metrics history to CSV. Defaults to True.

    Returns:
        dict: Training history containing per-epoch metric lists:
            - "student_loss": Mean total distillation loss per epoch.
            - "kd_loss": Mean KD loss component per epoch.
            - "spatial_loss": Mean spatial mapping loss component per epoch.
            - "relational_loss": Mean Barlow Twins relational loss component per epoch.
            - "student_lr": Student optimizer learning rate per epoch.
            - "epoch_time": Total elapsed time in seconds per epoch.
    """
    criterion = ODSDLoss(
        kd_temperature=kd_temperature,
        kd_modifier=kd_loss_modifier,
        spatial_modifier=spatial_loss_modifier,
        relational_modifier=relational_loss_modifier,
        barlow_twins_lambda=barlow_twins_lambda,
        huber_delta=huber_delta,
    ).to(accelerator.device)

    history = {
        "student_loss": [],
        "kd_loss": [],
        "spatial_loss": [],
        "relational_loss": [],
        "student_lr": [],
        "epoch_time": [],
    }

    teacher_model.eval()
    for epoch in range(epochs):
        student_model.train()
        total_train_loss = torch.tensor(0.0, device=accelerator.device)
        total_kd_loss = torch.tensor(0.0, device=accelerator.device)
        total_spatial_loss = torch.tensor(0.0, device=accelerator.device)
        total_relational_loss = torch.tensor(0.0, device=accelerator.device)
        train_total = torch.tensor(0, device=accelerator.device)

        start_time = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(memory_format=torch.channels_last)
            if augmentations is not None:
                augmented_images = augmentations(batch)
            if basic_augmentations is not None:
                basic_images = basic_augmentations(batch)

            batch_size = augmented_images.size(0)

            with accelerator.accumulate(student_model):
                with torch.no_grad():
                    teacher_output = teacher_model(augmented_images)  # type: ignore
                    teacher_basic_output = teacher_model(basic_images)

                student_output = student_model(augmented_images)  # type: ignore
                student_basic_output = student_model(basic_images)

                loss, loss_dict = criterion(
                    student_output=student_output,
                    teacher_output=teacher_output,
                    student_basic_output=student_basic_output,
                    teacher_basic_output=teacher_basic_output,
                    return_dict=True,
                )

                accelerator.backward(loss)

                if clipper is not None:
                    clipper(student_model.parameters())

                student_optimizer.step()
                student_optimizer.zero_grad()

                total_train_loss += loss.detach() * batch_size
                total_kd_loss += loss_dict["kd_loss"].detach() * batch_size
                total_spatial_loss += loss_dict["spatial_loss"].detach() * batch_size
                total_relational_loss += (
                    loss_dict["relational_loss"].detach() * batch_size
                )
                train_total += batch_size

        epoch_samples = accelerator.reduce(train_total, reduction="sum").item()  # type: ignore
        epoch_student_loss = (
            accelerator.reduce(total_train_loss, reduction="sum").item() / epoch_samples  # type: ignore
        )
        epoch_kd_loss = (
            accelerator.reduce(total_kd_loss, reduction="sum").item() / epoch_samples  # type: ignore
        )
        epoch_spatial_loss = (
            accelerator.reduce(total_spatial_loss, reduction="sum").item()  # type: ignore
            / epoch_samples
        )
        epoch_relational_loss = (
            accelerator.reduce(total_relational_loss, reduction="sum").item()  # type: ignore
            / epoch_samples
        )

        if scheduler is not None:
            scheduler.step()

        if accelerator.is_main_process:
            epoch_time = time.perf_counter() - start_time
            current_lr = student_optimizer.param_groups[0]["lr"]

            history["student_loss"].append(epoch_student_loss)
            history["kd_loss"].append(epoch_kd_loss)
            history["spatial_loss"].append(epoch_spatial_loss)
            history["relational_loss"].append(epoch_relational_loss)
            history["student_lr"].append(current_lr)
            history["epoch_time"].append(epoch_time)

            if epoch % print_frequency == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} Student Loss: {epoch_student_loss:.4f} "
                    f"(KD: {epoch_kd_loss:.4f}, Spatial: {epoch_spatial_loss:.4f}, Relational: {epoch_relational_loss:.4f}) | "
                    f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
                )

            if epoch % validation_frequency == 0 and validation_metrics is not None:
                eval_batch_size = validation_metrics.get("batch_size", 128)
                _ = evaluate_model(
                    model=student_model,
                    images=validation_metrics["validation_images"],
                    labels=validation_metrics["validation_labels"],
                    batch_size=eval_batch_size,
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
