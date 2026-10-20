import csv
import time
from pathlib import Path

import torch
from accelerate import Accelerator

from simple_kd.losses import KDLoss
from simple_kd.training import evaluate_model
from simple_kd.utils import SimpleGradClip

from .common import compute_generator_metrics, images_from_generator
from .losses import DAFLLoss


def dafl_train(accelerator: Accelerator,
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
               feature_gamma: float = 1.0,
               one_hot_gamma: float = 1.0,
               info_entropy_gamma: float = 128.0,
               kd_T: float = 4.0,
               kd_gamma: float = 1.0,
               validation_frequency: int = 50,
               scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
               generator_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
               teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...] = (),
               clipper: SimpleGradClip | None = None,
               validation_metrics: dict | None = None,
               save_location: Path | None = None,
               save_prefix: str = "",
               save_student_model: bool = True,
               save_generator: bool = False,
               save_history: bool = True) -> dict:
    """Trains a student model and generator using Data-Free Learning (DAFL).

    DAFL trains a generator to synthesize pseudo-data directly from a pre-trained teacher
    using three intrinsic losses, without requiring original data or real images:
    - Feature activation loss: Maximizes the intermediate layer activations of the teacher.
    - One-hot loss: Encourages confident, peaky class predictions for each image.
    - Information entropy loss: Maximizes the Shannon entropy across batch-averaged predictions,
      ensuring equal class distribution and preventing mode collapse.

    Concurrently, the student model is trained via Knowledge Distillation (KDLoss) to match
    the teacher's output distributions on these synthesized images in unified training steps.

    The original paper can be found at https://arxiv.org/pdf/1904.01186 (Chen et al., ICCV 2019).
    The original code implementation can be found at: https://github.com/autogyro/DAFL.

    Args:
        accelerator (Accelerator): The HuggingFace Accelerator instance.
        student_model (torch.nn.Module): The student model being trained.
        teacher_model (torch.nn.Module): The pre-trained frozen teacher model.
        generator (torch.nn.Module): Generative network mapping latent noise vectors to synthetic images.
        augmentations (torch.nn.Module): On-device augmentations and normalizations applied to generated images.
        student_optimizer (torch.optim.Optimizer): Optimizer for student model weights.
        generator_optimizer (torch.optim.Optimizer): Optimizer for generator network weights.
        epochs (int): Total number of training epochs.
        batch_size (int): Batch size of synthetic images generated per step.
        generator_latent_size (int): Dimensionality of the input latent noise vector z.
        steps_per_epoch (int, optional): Number of unified training steps per epoch. Defaults to 50.
        print_frequency (int, optional): Frequency (in epochs) to print progress to console. Defaults to 10.
        feature_gamma (float, optional): Multiplier for teacher feature activation loss. Defaults to 1.0.
        one_hot_gamma (float, optional): Multiplier for one-hot cross-entropy loss. Defaults to 1.0.
        info_entropy_gamma (float, optional): Multiplier for batch information entropy loss. Defaults to 128.0.
        kd_T (float, optional): Distillation temperature for student KDLoss. Defaults to 4.0.
        kd_gamma (float, optional): Multiplier for student KDLoss. Defaults to 1.0.
        validation_frequency (int, optional): Frequency (in epochs) to run validation evaluation. Defaults to 50.
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler for student. Defaults to None.
        generator_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler for generator. Defaults to None.
        teacher_stages (list[torch.Tensor] | tuple[torch.Tensor, ...], optional): Intermediate teacher activation hooks. Defaults to [].
        clipper (SimpleGradClip, optional): Gradient clipping utility applied before optimizer steps. Defaults to None.
        validation_metrics (dict, optional): Dictionary with validation images, labels, augs, and classes. Defaults to None.
        save_location (Path, optional): Directory where checkpoints and history CSV will be stored. Defaults to None.
        save_prefix (str, optional): Prefix for saved artifact filenames. Defaults to "".
        save_student_model (bool, optional): Whether to save student model checkpoint. Defaults to True.
        save_generator (bool, optional): Whether to save generator checkpoint. Defaults to False.
        save_history (bool, optional): Whether to save metrics history to CSV. Defaults to True.

    Returns:
        dict: Training history containing per-epoch metric lists:
            - "student_loss": Mean student KD loss per epoch.
            - "generator_loss": Mean generator DAFL loss per epoch.
            - "student_lr": Student optimizer learning rate per epoch.
            - "generator_lr": Generator optimizer learning rate per epoch.
            - "gen_conf": Mean confidence (max probability) of teacher on generator images.
            - "gen_entropy": Mean normalized class diversity entropy across batches.
            - "gen_spread": Mean spatial pixel standard deviation of generated images.
            - "epoch_time": Total elapsed time in seconds per epoch.
    """
    criterion = KDLoss(T=kd_T, gamma=kd_gamma).to(accelerator.device)
    generator_criterion = DAFLLoss(
        feature_gamma=feature_gamma,
        one_hot_gamma=one_hot_gamma,
        info_entropy_gamma=info_entropy_gamma
    ).to(accelerator.device)

    history = {
        "student_loss": [],
        "generator_loss": [],
        "student_lr": [],
        "generator_lr": [],
        "gen_conf": [],
        "gen_entropy": [],
        "gen_spread": [],
        "epoch_time": []
    }
    teacher_model.eval()

    for epoch in range(epochs):
        total_train_loss = torch.tensor(0.0, device=accelerator.device)
        total_gen_loss = torch.tensor(0.0, device=accelerator.device)
        gen_conf = torch.tensor(0.0, device=accelerator.device)
        gen_entropy = torch.tensor(0.0, device=accelerator.device)
        gen_spread = torch.tensor(0.0, device=accelerator.device)

        train_total = torch.tensor(0, device=accelerator.device)
        start_time = time.perf_counter()

        student_model.train()
        generator.train()
        for _ in range(steps_per_epoch):
            gen_imgs = images_from_generator(generator,augmentations,accelerator.device,batch_size,generator_latent_size,requires_grad=True)

            with accelerator.accumulate(student_model, generator):
                student_optimizer.zero_grad(set_to_none=True)
                generator_optimizer.zero_grad(set_to_none=True)

                teacher_output = teacher_model(gen_imgs) # Teacher stages are needed for training

                with torch.no_grad():
                    gen_statistics = compute_generator_metrics(gen_imgs, teacher_output)

                gen_loss = generator_criterion(
                    teacher_output=teacher_output,
                    teacher_stages=teacher_stages
                )

                student_output = student_model(gen_imgs.detach())
                student_loss = criterion(
                    student_output=student_output,
                    teacher_output=teacher_output.detach()
                )

                total_loss = gen_loss + student_loss
                accelerator.backward(total_loss)

                if clipper is not None:
                    accelerator.unscale_gradients(student_optimizer)
                    clipper(student_model.parameters())
                    accelerator.unscale_gradients(generator_optimizer)
                    clipper(generator.parameters())

                student_optimizer.step()
                generator_optimizer.step()

            total_train_loss += student_loss.detach() * batch_size
            total_gen_loss += gen_loss.detach() * batch_size
            train_total += batch_size
            gen_conf += gen_statistics["gen_conf"] * batch_size
            gen_entropy += gen_statistics["gen_entropy"] * batch_size
            gen_spread += gen_statistics["gen_spread"] * batch_size

        epoch_samples = accelerator.reduce(train_total, reduction="sum").item() # type: ignore
        model_epoch_loss = accelerator.reduce(total_train_loss, reduction="sum") / epoch_samples
        gen_epoch_loss = accelerator.reduce(total_gen_loss, reduction="sum") / epoch_samples
        gen_conf = accelerator.reduce(gen_conf, reduction="sum").item() / epoch_samples # type: ignore
        gen_entropy = accelerator.reduce(gen_entropy, reduction="sum").item() / epoch_samples # type: ignore
        gen_spread = accelerator.reduce(gen_spread, reduction="sum").item() / epoch_samples # type: ignore

        if scheduler is not None:
            scheduler.step()
        if generator_scheduler is not None:
            generator_scheduler.step()

        if accelerator.is_main_process:
            epoch_time = time.perf_counter() - start_time
            history["student_loss"].append(model_epoch_loss.item())
            history["student_lr"].append(student_optimizer.param_groups[0]['lr'])
            history["generator_loss"].append(gen_epoch_loss.item())
            history["generator_lr"].append(generator_optimizer.param_groups[0]['lr'])
            history["gen_conf"].append(gen_conf)
            history["gen_entropy"].append(gen_entropy)
            history["gen_spread"].append(gen_spread)
            history["epoch_time"].append(epoch_time)

            if epoch % print_frequency == 0:
                temp_message = f"Epoch {epoch+1}/{epochs} Student Train Loss: {model_epoch_loss:.4f}, Student lr: {student_optimizer.param_groups[0]['lr']:.6f},"
                temp_message += f" Generator confidence: {gen_conf:.4f}, Generator Entropy: {gen_entropy:.4f}, Generator spread: {gen_spread:.4f},"
                temp_message += f" Generator Train Loss: {gen_epoch_loss:.4f}, Generator lr: {generator_optimizer.param_groups[0]['lr']:.6f}, Training time: {epoch_time}"
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
                    save_folder=None
                )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and save_location is not None:
        if save_student_model:
            student_model_name = f"{save_prefix}_student_model.pth" if save_prefix else "student_model.pth"
            student_model_savepath = save_location / student_model_name
            student_model_savepath.parent.mkdir(parents=True, exist_ok=True)
            unwrapped_student_model = accelerator.unwrap_model(student_model, keep_torch_compile=False)
            torch.save(unwrapped_student_model.state_dict(), student_model_savepath)

        if save_generator:
            generator_name = f"{save_prefix}_generator.pth" if save_prefix else "generator.pth"
            generator_savepath = save_location / generator_name
            generator_savepath.parent.mkdir(parents=True, exist_ok=True)
            unwrapped_generator = accelerator.unwrap_model(generator, keep_torch_compile=False)
            torch.save(unwrapped_generator.state_dict(), generator_savepath)

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
