import csv
import time
from pathlib import Path

import torch
from accelerate import Accelerator

from simple_kd.utils import SimpleGradClip


def train(
    accelerator: Accelerator,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int = 10,
    epochs_per_val: int = 1,
    clipper: SimpleGradClip | None = None,
    augmentations: torch.nn.Module | None = None,
    val_augmentations: torch.nn.Module | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    val_loader: torch.utils.data.DataLoader | None = None,
    save_location: Path | None = None,
    save_prefix: str = "",
    save_model: bool = True,
    save_history: bool = True,
) -> dict:
    """A concise training function for a neural model, configured primarily around classification tasks.

    Args:
        accelerator (Accelerator): The HuggingFace Accelerator instance managing device placement and precision.
        model (torch.nn.Module): The model that will be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader providing training batches.
        optimizer (torch.optim.Optimizer): The optimizer used for model updates.
        criterion (torch.nn.Module): The loss criterion used for training (e.g. CrossEntropyLoss).
        epochs (int, optional): The total number of training epochs. Defaults to 10.
        epochs_per_val (int, optional): Evaluation frequency. If set > 1, skips intermediate validation epochs for speed. Defaults to 1.
        clipper (SimpleGradClip, optional): Optional gradient clipper applied before optimizer step. Defaults to None.
        augmentations (torch.nn.Module, optional): On-device training augmentations. Defaults to None.
        val_augmentations (torch.nn.Module, optional): On-device validation augmentations. Defaults to None.
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler stepped once per epoch. Defaults to None.
        val_loader (torch.utils.data.DataLoader, optional): The DataLoader providing validation batches. Defaults to None.
        save_location (Path, optional): Directory path where checkpoints and history CSV will be saved. Defaults to None.
        save_prefix (str, optional): Optional prefix prepended to saved checkpoint and history filenames. Defaults to "".
        save_model (bool, optional): Whether to save the unwrapped model state dictionary. Defaults to True.
        save_history (bool, optional): Whether to save training metrics history as a CSV file. Defaults to True.

    Returns:
        dict: A dictionary containing per-epoch metric lists:
            - "train_loss": Average training loss per epoch.
            - "train_accuracy": Training accuracy percentage per epoch.
            - "val_loss": Validation loss per epoch (or repeated from previous if skipped).
            - "val_accuracy": Validation accuracy percentage per epoch.
            - "lr": Current learning rate per epoch.
            - "epoch_time": Total elapsed time in seconds per epoch.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "lr": [],
        "epoch_time": [],
    }

    for epoch in range(epochs):
        train_loss = torch.tensor(0.0, device=accelerator.device)
        train_correct = torch.tensor(0, device=accelerator.device)
        train_total = torch.tensor(0, device=accelerator.device)

        temp_message = f"Start of epoch {epoch + 1}"
        start_time = time.perf_counter()
        model.train()
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

            with accelerator.accumulate(model):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                accelerator.backward(loss)

                if clipper is not None and accelerator.sync_gradients:
                    accelerator.unscale_gradients(optimizer)
                    clipper(model.parameters())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Compute training loss and accuracy
            _, predicted = torch.max(
                outputs, 1
            )  # Get the index of the max logit (predicted class)
            current_batch_size = labels.size(0)
            train_loss += loss.detach() * current_batch_size
            train_correct += (predicted == labels).sum()
            train_total += current_batch_size

        epoch_train_loss = accelerator.reduce(train_loss, reduction="sum").item()  # type: ignore
        epoch_train_correct = accelerator.reduce(train_correct, reduction="sum").item()  # type: ignore
        epoch_train_samples = accelerator.reduce(train_total, reduction="sum").item()  # type: ignore

        # Final epoch statistics
        epoch_train_loss = epoch_train_loss / epoch_train_samples
        epoch_train_accuracy = 100.0 * epoch_train_correct / epoch_train_samples

        # Validation Phase
        epoch_val_loss = torch.tensor(0.0)
        epoch_val_accuracy = torch.tensor(0.0)
        if val_loader is not None and epoch % epochs_per_val == 0:
            model.eval()
            val_loss = torch.tensor(0.0, device=accelerator.device)
            val_correct = torch.tensor(0, device=accelerator.device)
            val_total = torch.tensor(0, device=accelerator.device)

            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs = inputs.to(memory_format=torch.channels_last)

                    if val_augmentations is not None:
                        inputs, labels = val_augmentations(inputs, labels)

                    if labels.ndim > 1:
                        labels = labels.argmax(dim=-1).squeeze().long()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Compute val loss and accuracy
                    _, predicted = torch.max(outputs, 1)
                    current_batch_size = torch.tensor(
                        labels.size(0), device=accelerator.device
                    )
                    val_loss += loss.detach() * current_batch_size
                    val_correct += (predicted == labels).sum()
                    val_total += current_batch_size

                epoch_val_loss = accelerator.reduce(val_loss, reduction="sum").item()  # type: ignore
                epoch_val_correct = accelerator.reduce(
                    val_correct, reduction="sum"
                ).item()  # type: ignore
                epoch_val_samples = accelerator.reduce(
                    val_total, reduction="sum"
                ).item()  # type: ignore

                epoch_val_loss = epoch_val_loss / epoch_val_samples
                epoch_val_accuracy = 100.0 * epoch_val_correct / epoch_val_samples

        if scheduler is not None:
            scheduler.step()

        if accelerator.is_main_process:
            epoch_time = time.perf_counter() - start_time
            # Print both loss and accuracy at the end of each epoch
            temp_message = f"Epoch {epoch + 1}/{epochs} Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%,"
            if val_loader is not None:
                if epoch % epochs_per_val == 0:
                    temp_message += f" Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%,"
                    history["val_accuracy"].append(epoch_val_accuracy)
                    history["val_loss"].append(epoch_val_loss)
                else:
                    history["val_accuracy"].append(history["val_accuracy"][-1])
                    history["val_loss"].append(history["val_loss"][-1])
            temp_message += f"Training time: {epoch_time:.2f} Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"

            history["train_accuracy"].append(epoch_train_accuracy)
            history["train_loss"].append(epoch_train_loss)
            history["epoch_time"].append(epoch_time)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            print(temp_message)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and save_location is not None:
        if save_model:
            model_name = f"{save_prefix}_model.pth" if save_prefix else "model.pth"
            model_savepath = save_location / model_name
            model_savepath.parent.mkdir(parents=True, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model, keep_torch_compile=False)
            torch.save(unwrapped_model.state_dict(), model_savepath)

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
