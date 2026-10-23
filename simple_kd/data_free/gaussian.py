import csv
import time
from pathlib import Path

import torch
from accelerate import Accelerator

from simple_kd.losses import KDLoss
from simple_kd.training import evaluate_model
from simple_kd.utils import SimpleGradClip


def gaussian_train(accelerator: Accelerator,
                student_model: torch.nn.Module,
                teacher_model: torch.nn.Module,
                augmentations: torch.nn.Module,
                student_optimizer: torch.optim.Optimizer,
                epochs: int,
                batch_size: int,
                synthetic_loader: torch.utils.data.DataLoader | None = None,
                steps_per_epoch: int = 50,
                noise_shape: tuple[int, int, int] = (3, 32, 32),
                print_frequency: int = 10,
                model_multipler: float = 1.0,
                temperature: float = 4.0,
                validation_frequency: int = 50,
                scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                clipper: SimpleGradClip | None = None,
                validation_metrics: dict | None = None,
                save_location: Path | None = None,
                save_prefix: str = "",
                save_student_model: bool = True,
                save_history: bool = True) -> dict:
    """This is a custom implementation of the method described from the paper Discovering and Overcoming Limitations of
    Noise-engineered Data-free Knowledge Distillation (Raikwar, Mishra). This method utilizes active batch norms to learn from gaussian noise.

    This version of the method can utilize a validation loader in order to use a synthetic noise dataset, such as the dead leaves dataset or stylegan.

    The original paper can be found at https://proceedings.neurips.cc/paper_files/paper/2022/file/1f96b24df4b06f5d68389845a9a13ed9-Paper-Conference.pdf.

    The original code implementation can be found at https://github.com/Piyush-555/GaussianDistillation.

    Args:
        accelerator (Accelerator): The HuggingFace Accelerator instance.
        student_model (torch.nn.Module): The student model to be trained.
        teacher_model (torch.nn.Module): The pre-trained frozen teacher model.
        augmentations (torch.nn.Module): On-device augmentations and normalizations applied to generated images. NO augmentations that require labels such as mixup.
        student_optimizer (torch.optim.Optimizer): Optimizer for student model weights.
        epochs (int): Total number of training epochs.
        batch_size (int): Batch size of synthetic images per step.
        steps_per_epoch (int, optional): Number of student training steps per epoch. Defaults to 50. Ignored when using synthetic loader.
        noise_shape (tuple[int, int, int], optional): Noise shape of gaussian noise, channels first. Ignored when using synthetic loader. default=(3, 32, 32),
        synthetic_loader (torch.utils.data.DataLoader | None, optional): Torch dataloader with synthetic noise to use instead of pure gaussian noise.
        print_frequency (int, optional): Frequency (in epochs) to print progress to console. Defaults to 10.
        model_multipler (float, optional): Multiplier/gamma for the student DFAD loss. Defaults to 1.0.
        temperature (float, optional): The KD loss temperature. Defaults to 4.0.
        validation_frequency (int, optional): Frequency (in epochs) to evaluate on validation metrics. Defaults to 50.
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler for student. Defaults to None.
        clipper (SimpleGradClip, optional): Gradient clipping utility. Defaults to None.
        validation_metrics (dict, optional): Dictionary with validation images, labels, augs, and classes for periodic evaluation. Defaults to None.
        save_location (Path, optional): Directory where checkpoints and history CSV will be stored. Defaults to None.
        save_prefix (str, optional): Prefix for saved artifact filenames. Defaults to "".
        save_student_model (bool, optional): Whether to save student model checkpoint. Defaults to True.
        save_history (bool, optional): Whether to save metrics history to CSV. Defaults to True.

    Returns:
        dict: Training history containing per-epoch metric lists:
            - "student_loss": Mean student discrepancy loss per epoch.
            - "student_lr": Student optimizer learning rate per epoch.
            - "epoch_time": Total elapsed time in seconds per epoch.
    """
    criterion = KDLoss(T=temperature, gamma=model_multipler)
    history = {"student_loss": [], "student_lr": [], "epoch_time": []}
    teacher_model.train() # While frozen, teacher needs to be on train mode to update batch statistics. Quirk unique to this method!
    if synthetic_loader is None:
        z = torch.empty(
            (batch_size, noise_shape[0], noise_shape[1], noise_shape[2]),
            device=accelerator.device,
            dtype=next(student_model.parameters()).dtype,
            memory_format=torch.channels_last,
        )
    else:
        z = torch.tensor(0.0) # Completely unused, this exists to calm the linters

    for epoch in range(epochs):
        student_model.train()
        total_train_loss = torch.tensor(0.0, device=accelerator.device)
        train_total = torch.tensor(0, device=accelerator.device)
        start_time = time.perf_counter()
        if synthetic_loader is not None:
            for batch in synthetic_loader:
                batch = batch.to(memory_format=torch.channels_last)

                if augmentations is not None:
                    batch = augmentations(batch)

                if not batch.is_contiguous(memory_format=torch.channels_last):
                    batch = batch.contiguous(memory_format=torch.channels_last)

                with accelerator.accumulate(student_model):
                    student_output = student_model(batch)
                    with torch.no_grad():
                        teacher_output = teacher_model(batch)
                    train_loss = criterion(student_output, teacher_output)
                    accelerator.backward(train_loss)

                    total_train_loss += train_loss.detach() * batch_size
                    train_total += batch_size

                    if clipper is not None and accelerator.sync_gradients:
                        accelerator.unscale_gradients(student_optimizer)
                        clipper(student_model.parameters())
                    student_optimizer.step()
                    student_optimizer.zero_grad(set_to_none=True)
        else:
            for _ in range(steps_per_epoch):
                with accelerator.accumulate(student_model):
                    z.uniform_()
                    if augmentations is not None:
                        z = augmentations(z)
                    student_output = student_model(z)
                    with torch.no_grad():
                        teacher_output = teacher_model(z)

                    train_loss = criterion(student_output, teacher_output)
                    accelerator.backward(train_loss)
                    total_train_loss += train_loss.detach() * batch_size
                    train_total += batch_size

                    if clipper is not None and accelerator.sync_gradients:
                        accelerator.unscale_gradients(student_optimizer)
                        clipper(student_model.parameters())
                    student_optimizer.step()
                    student_optimizer.zero_grad(set_to_none=True)

        epoch_samples = accelerator.reduce(train_total, reduction="sum").item() # type: ignore
        model_epoch_loss = accelerator.reduce(total_train_loss, reduction="sum") / epoch_samples

        if scheduler is not None:
            scheduler.step()

        if accelerator.is_main_process:
            epoch_time = time.perf_counter() - start_time
            history["student_loss"].append(model_epoch_loss.item())
            history["student_lr"].append(student_optimizer.param_groups[0]['lr'])
            history["epoch_time"].append(epoch_time)

            if epoch % print_frequency == 0:
                temp_message = f"Epoch {epoch+1}/{epochs} Student Train Loss: {model_epoch_loss:.4f}, Student lr: {student_optimizer.param_groups[0]['lr']:.6f}, Training time: {epoch_time}"
                print(temp_message)

            if epoch % validation_frequency == 0 and validation_metrics is not None:
                _ = evaluate_model(model=student_model,
                    images=validation_metrics["validation_images"],
                    labels=validation_metrics["validation_labels"],
                    batch_size=batch_size,
                    device=accelerator.device,
                    augs=validation_metrics["validation_augs"],
                    criterion=torch.nn.CrossEntropyLoss(),
                    classes=validation_metrics["classes"],
                    save_folder=None)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and save_location is not None:
        if save_student_model:
            student_model_name = f"{save_prefix}_student_model.pth" if save_prefix else "student_model.pth"
            student_model_savepath = save_location / student_model_name
            student_model_savepath.parent.mkdir(parents=True, exist_ok=True)
            unwrapped_student_model = accelerator.unwrap_model(student_model, keep_torch_compile=False)
            torch.save(unwrapped_student_model.state_dict(), student_model_savepath)

        if save_history:
            history_name = f"{save_prefix}_history.csv" if save_prefix else "history.csv"
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
