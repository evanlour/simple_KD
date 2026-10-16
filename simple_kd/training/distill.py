import csv
import inspect
import time
import torch
from accelerate import Accelerator
from simple_kd.utils import SimpleGradClip
from pathlib import Path

def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    """Passes only kwargs that fn's signature actually accepts, ignoring None."""
    target = getattr(fn, "forward", fn)
    sig = inspect.signature(target)
    valid_names = set(sig.parameters.keys())
    return {
        k: v for k, v in kwargs.items()
        if k in valid_names and v is not None
    }

def distill(accelerator: Accelerator,
          teacher_model: torch.nn.Module,
          student_model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          student_optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          distillation_criterion: torch.nn.Module,
          student_stages: list[torch.Tensor] | tuple[torch.Tensor, ...] =(),
          teacher_stages: list[torch.Tensor] | tuple[torch.Tensor, ...] =(),
          teacher_optimizer: torch.optim.Optimizer | None = None,
          epochs: int = 10,
          epochs_per_val: int = 1,
          clipper: SimpleGradClip | None = None,
          augmentations: torch.nn.Module | None=None,
          val_augmentations: torch.nn.Module | None=None,
          teacher_scheduler: torch.optim.lr_scheduler.LRScheduler | None=None,
          student_scheduler: torch.optim.lr_scheduler.LRScheduler | None=None,
          val_loader: torch.utils.data.DataLoader | None=None,
          save_location: Path | None = None,
          save_prefix: str = "",
          save_student_model: bool = True,
          save_teacher_model: bool = False,
          save_history: bool = True) -> dict:
    """ This function distills knowledge from a teacher model to a student model. Supports multiple options, is configured around classification tasks.

    Args:
        accelerator (Accelerator): The accelerator instance.
        teacher_model (torch.nn.Module): The model that will be used as a teacher.
        student_model (torch.nn.Module): The model that will be used as a student.
        train_loader (torch.utils.data.DataLoader): The train data that will be used.
        student_optimizer (torch.optim): The optimizer that will be used for the student.
        criterion (torch.nn.Module): The non distillation criterion that will be used for the teacher and testing, usually CrossEntropy().
        distillation_criterion (torch.nn.Module): The function that will be used for calculating the distillation loss.
        student_stages(list[torch.Tensor]): List of tensors that derive from forward hooks of the student model. Defaults to None.
        teacher_stages(list[torch.Tensor]): List of tensors that derive from forward hooks of the teacher model. Defaults to None.
        teacher_optimizer (torch.optim.Optimizer, optional): Set an optimizer if you want to train the teacher as well ONLY with criterion. Defaults to None.
        epochs (int, optional): The number of epochs the distillation will take place. Defaults to 10.
        epochs_per_val (int, optional): If set > 1 it will skip validation epochs, in case speed is needed. Defaults to 1.
        clipper (SimpleGradClip, optional): If not none, clips the gradients. Defaults to None.
        augmentations (torch.nn.Module, optional): The training augmentations. These happen on device. Defaults to None.
        val_augmentations (torch.nn.Module, optional): The validation augmentations. These happen on device. Defaults to None.
        teacher_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): The student learning rate scheduler. Called once per epoch. Defaults to None.
        student_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): The teacher learning rate scheduler. Called once per epoch. Defaults to None.. Defaults to None.
        val_loader (torch.utils.data.DataLoader, optional): The validation data that will be used. Defaults to None.
        save_location (str, optional): If not none, sets the location where the history and the model will be saved. Defaults to None.
        save_student_model (bool, optional): Dictates whether the student model will be saved. Defaults to True.
        save_teacher_model (bool, optional): Dictates whether the teacher model will be saved. Defaults to False.
        save_history (bool, optional): Dictates whether the history will be saved. Defaults to True.

    Returns:
        history(dict): Per epoch statistics around train and validation accuracy, along with the learning rate and the time of each epoch.

        If on an epoch the validation step is skipped, the data written are from the previous epoch.
    """
    history = {"student_train_loss": [], "student_train_accuracy": [], "teacher_train_loss": [], "teacher_train_accuracy": [],
        "student_val_loss": [], "student_val_accuracy": [], "teacher_val_loss": [], "teacher_val_accuracy": [],
        "lr": [], "epoch_time": []}

    for epoch in range (epochs):
        student_train_loss = torch.tensor(0.0, device=accelerator.device)
        student_train_correct = torch.tensor(0, device=accelerator.device)
        train_total = torch.tensor(0, device=accelerator.device)
        teacher_train_loss = torch.tensor(0.0, device=accelerator.device)
        teacher_train_correct = torch.tensor(0, device=accelerator.device)

        temp_message = f'Start of epoch {epoch + 1}'
        start_time = time.perf_counter()
        student_model.train()
        if teacher_optimizer is not None: # Online distillation when teacher optimizer is used
            teacher_model.train()
        else:
            teacher_model.eval()

        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(memory_format=torch.channels_last)
            labels = labels.view(-1)
            if augmentations is not None:
                inputs, labels = augmentations(inputs, labels)

            if not inputs.is_contiguous(memory_format=torch.channels_last):
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            if labels.ndim > 1:
                labels = labels.argmax(dim=-1).squeeze().long()

            models_to_accumulate = [student_model]
            if teacher_optimizer is not None:
                models_to_accumulate.append(teacher_model)

            with accelerator.accumulate(*models_to_accumulate):
                if teacher_optimizer is not None:
                    teacher_output = teacher_model(inputs)
                else:
                    with torch.no_grad():
                        teacher_output = teacher_model(inputs)

                student_output = student_model(inputs)

                teacher_loss = criterion(teacher_output, labels)
                student_ce_loss = criterion(student_output, labels)

                available_args = {
                    "student_output": student_output,
                    "teacher_output": teacher_output.detach() if teacher_output is not None else None,
                    "student_stages": student_stages,
                    "teacher_stages": [v.detach() for v in teacher_stages] if teacher_stages is not None else None,
                    "labels": labels,
                    "images": inputs,
                }
                filtered_args = _filter_kwargs_for_callable(distillation_criterion, available_args)
                distill_loss = distillation_criterion(**filtered_args)

                student_loss = distill_loss + student_ce_loss
                total_loss = teacher_loss + student_loss
                accelerator.backward(total_loss)

                if clipper is not None and accelerator.sync_gradients:
                    accelerator.unscale_gradients(student_optimizer)
                    clipper(student_model.parameters())
                    if teacher_optimizer is not None:
                        accelerator.unscale_gradients(teacher_optimizer)
                        clipper(teacher_model.parameters())

            student_optimizer.step()
            if teacher_optimizer is not None:
                teacher_optimizer.step()

            student_optimizer.zero_grad(set_to_none=True)
            if teacher_optimizer is not None:
                teacher_optimizer.zero_grad(set_to_none=True)

            current_batch_size = labels.size(0)
            train_total += current_batch_size
            # student statistics
            _, student_predicted = torch.max(student_output, 1)
            student_train_loss += student_loss.detach() * current_batch_size
            student_train_correct += (student_predicted == labels).sum()
            # teacher statistics
            _, teacher_predicted = torch.max(teacher_output, 1) # type: ignore
            teacher_train_loss += teacher_loss.detach() * current_batch_size
            teacher_train_correct += (teacher_predicted == labels).sum()

        epoch_train_samples = accelerator.reduce(train_total, reduction="sum").item()   # type: ignore
        # Final student statistics
        student_epoch_train_loss = accelerator.reduce(student_train_loss, reduction="sum").item()    # type: ignore
        student_epoch_train_loss = student_epoch_train_loss / epoch_train_samples
        student_epoch_train_correct = accelerator.reduce(student_train_correct, reduction="sum").item() # type: ignore
        student_epoch_train_accuracy = 100.0 * student_epoch_train_correct / epoch_train_samples # type: ignore
        # Final teacher statistics
        teacher_epoch_train_loss    = accelerator.reduce(teacher_train_loss, reduction="sum").item()    # type: ignore
        teacher_epoch_train_loss = teacher_epoch_train_loss / epoch_train_samples
        teacher_epoch_train_correct = accelerator.reduce(teacher_train_correct, reduction="sum").item() # type: ignore
        teacher_epoch_train_accuracy = 100.0 * teacher_epoch_train_correct / epoch_train_samples # type: ignore

        # Validation Phase
        student_epoch_val_loss = torch.tensor(0.0)
        student_epoch_val_accuracy = torch.tensor(0.0)
        teacher_epoch_val_loss = torch.tensor(0.0)
        teacher_epoch_val_accuracy = torch.tensor(0.0)
        if val_loader is not None and epoch % epochs_per_val == 0:
            student_model.eval()
            teacher_model.eval()
            val_total = torch.tensor(0, device=accelerator.device)
            student_val_loss = torch.tensor(0.0, device=accelerator.device)
            student_val_correct = torch.tensor(0, device=accelerator.device)
            teacher_val_loss = torch.tensor(0.0, device=accelerator.device)
            teacher_val_correct = torch.tensor(0, device=accelerator.device)

            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs = inputs.to(memory_format=torch.channels_last)

                    if val_augmentations is not None:
                        inputs, labels = val_augmentations(inputs, labels)

                    if labels.ndim > 1:
                        labels = labels.argmax(dim=-1).squeeze().long()

                    student_output = student_model(inputs)
                    teacher_output = teacher_model(inputs)
                    student_loss = criterion(student_output, labels)
                    teacher_loss = criterion(teacher_output, labels)

                    current_batch_size = labels.size(0)
                    val_total += current_batch_size
                    # student statistics
                    _, student_predicted = torch.max(student_output, 1)
                    student_val_loss += student_loss * current_batch_size
                    student_val_correct += (student_predicted == labels).sum()
                    # teacher statistics
                    _, teacher_predicted = torch.max(teacher_output, 1)
                    teacher_val_loss += teacher_loss * current_batch_size
                    teacher_val_correct += (teacher_predicted == labels).sum()

                epoch_val_samples = accelerator.reduce(val_total, reduction="sum").item()   # type: ignore
                # Final student statistics
                student_epoch_val_loss = accelerator.reduce(student_val_loss, reduction="sum").item()    # type: ignore
                student_epoch_val_loss = student_epoch_val_loss / epoch_val_samples
                student_epoch_val_correct = accelerator.reduce(student_val_correct, reduction="sum").item() # type: ignore
                student_epoch_val_accuracy = 100.0 * student_epoch_val_correct / epoch_val_samples # type: ignore
                # Final teacher statistics
                teacher_epoch_val_loss    = accelerator.reduce(teacher_val_loss, reduction="sum").item()    # type: ignore
                teacher_epoch_val_loss = teacher_epoch_val_loss / epoch_val_samples
                teacher_epoch_val_correct = accelerator.reduce(teacher_val_correct, reduction="sum").item() # type: ignore
                teacher_epoch_val_accuracy = 100.0 * teacher_epoch_val_correct / epoch_val_samples # type: ignore

        if student_scheduler is not None:
            student_scheduler.step()
        if teacher_scheduler is not None:
            teacher_scheduler.step()

        if accelerator.is_main_process:
            epoch_time = time.perf_counter() - start_time
            # Print both loss and accuracy at the end of each epoch
            temp_message = f"Epoch {epoch+1}/{epochs} Student Train Loss: {student_epoch_train_loss:.4f}, Student Train Acc: {student_epoch_train_accuracy:.2f}%,"
            temp_message += f" Teacher Train Loss: {teacher_epoch_train_loss:.4f}, Teacher Train Acc: {teacher_epoch_train_accuracy:.2f}%,"
            if val_loader is not None:
                if epoch % epochs_per_val == 0:
                    temp_message += f" Student Val Loss: {student_epoch_val_loss:.4f}, Val Acc: {student_epoch_val_accuracy:.2f}%,"
                    temp_message += f" Teacher Val Loss: {teacher_epoch_val_loss:.4f}, Val Acc: {teacher_epoch_val_accuracy:.2f}%,"
                    history["student_val_accuracy"].append(student_epoch_val_accuracy)
                    history["teacher_val_accuracy"].append(teacher_epoch_val_accuracy)
                    history["student_val_loss"].append(student_epoch_val_loss)
                    history["teacher_val_loss"].append(teacher_epoch_val_loss)
                else:
                    history["student_val_accuracy"].append(history["student_val_accuracy"][-1])
                    history["student_val_loss"].append(history["student_val_loss"][-1])
                    history["teacher_val_accuracy"].append(history["teacher_val_accuracy"][-1])
                    history["teacher_val_loss"].append(history["teacher_val_loss"][-1])
            temp_message += f"Training time: {epoch_time:.2f} Current Student Learning Rate: {student_optimizer.param_groups[0]['lr']:.6f}"

            history["student_train_accuracy"].append(student_epoch_train_accuracy)
            history["student_train_loss"].append(student_epoch_train_loss)
            history["teacher_train_accuracy"].append(teacher_epoch_train_accuracy)
            history["teacher_train_loss"].append(teacher_epoch_train_loss)
            history["epoch_time"].append(epoch_time)
            history["lr"].append(student_optimizer.param_groups[0]['lr'])
            print(temp_message)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if save_location is not None:
            if save_student_model:
                student_model_name = f"{save_prefix}_student_model.pth" if save_prefix else "student_model.pth"
                student_model_savepath = save_location / student_model_name
                student_model_savepath.parent.mkdir(parents=True, exist_ok=True)
                unwrapped_student_model = accelerator.unwrap_model(student_model, keep_torch_compile=False)
                torch.save(unwrapped_student_model.state_dict(), student_model_savepath)

            if save_teacher_model:
                teacher_model_name = f"{save_prefix}_teacher_model.pth" if save_prefix else "teacher_model.pth"
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
