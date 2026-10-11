import time
import torch
from accelerate import Accelerator
from .common import images_from_generator, train_model_with_generator, train_generator
from simple_kd.utils import SimpleGradClip
from .losses import DFADLoss
from simple_kd.training import evaluate_model
from pathlib import Path
import csv

def dfad_train(accelerator: Accelerator,
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
                validation_frequency: int = 50,
                scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                generator_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                student_stages: list[torch.Tensor] | tuple[torch.Tensor] =[],
                teacher_stages: list[torch.Tensor] | tuple[torch.Tensor] =[],
                clipper: SimpleGradClip | None = None,
                validation_metrics: dict | None = None,
                save_location: Path | None = None,
                save_prefix: str = "",
                save_student_model: bool = True,
                save_generator: bool = False,
                save_history: bool = True) -> dict:

    criterion = DFADLoss(use_log1p=True).to(accelerator.device)
    generator_criterion = DFADLoss(sign=-1, use_log1p=True).to(accelerator.device)
    history = {"student_loss": [], "generator_loss": [], "student_lr": [], "generator_lr": [], "epoch_time": []}
    teacher_model.eval()
    for epoch in range(epochs):
        total_train_loss = torch.tensor(0.0, device=accelerator.device)
        train_total = torch.tensor(0, device=accelerator.device)
        start_time = time.perf_counter()
        for _ in range(steps_per_epoch):
            gen_imgs = images_from_generator(generator, augmentations, accelerator.device, batch_size, generator_latent_size)
            gen_imgs = gen_imgs.to(memory_format=torch.channels_last)
            train_loss = train_model_with_generator(accelerator, gen_imgs, student_model, teacher_model,
                generator, criterion, student_optimizer, student_stages=student_stages, teacher_stages=teacher_stages, clipper=clipper)
            total_train_loss += train_loss.detach() * batch_size
            train_total += batch_size

        epoch_samples = accelerator.reduce(train_total, reduction="sum").item() # type: ignore
        model_epoch_loss = accelerator.reduce(total_train_loss, reduction="sum")
        model_epoch_loss = model_epoch_loss / epoch_samples

        gen_imgs = images_from_generator(generator, augmentations, accelerator.device, batch_size, generator_latent_size, requires_grad=True)
        gen_loss = train_generator(accelerator, gen_imgs, student_model, teacher_model,
            generator, generator_criterion, generator_optimizer, student_stages=student_stages, teacher_stages=teacher_stages, clipper=clipper)

        if scheduler is not None:
            scheduler.step()
        if generator_scheduler is not None:
            generator_scheduler.step()

        if accelerator.is_main_process:
            epoch_time = time.perf_counter() - start_time
            history["student_loss"].append(model_epoch_loss.item())
            history["student_lr"].append(student_optimizer.param_groups[0]['lr'])
            history["generator_loss"].append(gen_loss.item())
            history["generator_lr"].append(generator_optimizer.param_groups[0]['lr'])
            history["epoch_time"].append(epoch_time)

            if epoch % print_frequency == 0:
                temp_message = f"Epoch {epoch+1}/{epochs} Student Train Loss: {model_epoch_loss:.4f}, Student lr: {student_optimizer.param_groups[0]['lr']:.6f},"
                temp_message += f" Generator Train Loss: {gen_loss:.4f}, Generator lr: {generator_optimizer.param_groups[0]['lr']:.6f}, Training time: {epoch_time}"
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
    if accelerator.is_main_process:
        if save_location is not None:
            if save_student_model:
                student_model_name = f"{save_prefix}_student_model.pth" if save_prefix else "student_model.pth"
                student_model_savepath = save_location / student_model_name
                student_model_savepath.parent.mkdir(parents=True, exist_ok=True)
                unwrapped_student_model = accelerator.unwrap_model(student_model, keep_torch_compile=False)
                torch.save(unwrapped_student_model.state_dict(), student_model_savepath)

            if save_generator:
                teacher_model_name = f"{save_prefix}_generator.pth" if save_prefix else "generator.pth"
                teacher_model_savepath = save_location / teacher_model_name
                teacher_model_savepath.parent.mkdir(parents=True, exist_ok=True)
                unwrapped_teacher_model = accelerator.unwrap_model(teacher_model, keep_torch_compile=False)
                torch.save(unwrapped_teacher_model.state_dict(), teacher_model_savepath)

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
