import csv
import time
import torch
from accelerate import Accelerator
from simple_kd.utils import SimpleGradClip
from pathlib import Path

def train(accelerator: Accelerator,
          model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          epochs: int = 10,
          epochs_per_val: int = 1,
          clipper: SimpleGradClip | None = None,
          augmentations: torch.nn.Module | None =None,
          val_augmentations: torch.nn.Module | None =None,
          scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
          val_loader: torch.utils.data.DataLoader | None =None,
          save_location: Path | None = None,
          save_prefix: str = "",
          save_model: bool = True,
          save_history: bool = True) -> dict:
    """A concise train function for a neural model. Configured mostly around classification tasks.

    Args:
        accelerator (Accelerator): The accelerator instance.
        model (torch.nn.Module): The model that will be trained.
        train_loader (torch.utils.data.DataLoader): The train data that will be used.
        optimizer (torch.optim): The optimizer that will be used.
        criterion (torch.nn.Module): The criterion which the training will take place.
        epochs (int, optional): The training epochs. Defaults to 10.
        epochs_per_val (int, optional): If set > 1 it will skip validation epochs, in case speed is needed. Defaults to 1.
        clipper (SimpleGradClip, optional): If not none, clips the gradients. Defaults to None.
        augmentations (torch.nn.Module, optional): The training augmentations. These happen on device. Defaults to None.
        val_augmentations (torch.nn.Module, optional): The validation augmentations. These happen on device. Defaults to None.
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler. Called once per epoch. Defaults to None.
        val_loader (torch.utils.data.DataLoader, optional): The validation data that will be used. Defaults to None.
        save_location (str, optional): If not none, sets the location where the history and the model will be saved. Defaults to None.
        save_model (bool, optional): Dictates whether the model will be saved. Defaults to True.
        save_history (bool, optional): Dictates whether the history will be saved. Defaults to True.

    Returns:
        history(dict): Per epoch statistics around train and validation accuracy, along with the learning rate and the time of each epoch.

        If on an epoch the validation step is skipped, the data written are from the previous epoch.
    """

    history = {"train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "lr": [], "epoch_time": []}

    for epoch in range (epochs):
        train_loss = torch.tensor(0.0, device=accelerator.device)
        train_correct = torch.tensor(0, device=accelerator.device)
        train_total = torch.tensor(0, device=accelerator.device)

        temp_message = f'Start of epoch {epoch + 1}'
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
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                accelerator.backward(loss)

                if clipper is not None:
                    accelerator.unscale_gradients(optimizer)
                    clipper(model.parameters())
                optimizer.step()

            # Compute training loss and accuracy
            _, predicted = torch.max(outputs, 1)  # Get the index of the max logit (predicted class)
            current_batch_size = labels.size(0)
            train_loss += loss.detach() * current_batch_size
            train_correct += (predicted == labels).sum()
            train_total += current_batch_size

        epoch_train_loss    = accelerator.reduce(train_loss, reduction="sum").item()    # type: ignore
        epoch_train_correct = accelerator.reduce(train_correct, reduction="sum").item() # type: ignore
        epoch_train_samples = accelerator.reduce(train_total, reduction="sum").item()   # type: ignore

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
                    current_batch_size = torch.tensor(labels.size(0), device=accelerator.device)
                    val_loss += loss.detach() * current_batch_size
                    val_correct += (predicted == labels).sum()
                    val_total += current_batch_size

                epoch_val_loss    = accelerator.reduce(val_loss, reduction="sum").item()    # type: ignore
                epoch_val_correct = accelerator.reduce(val_correct, reduction="sum").item() # type: ignore
                epoch_val_samples = accelerator.reduce(val_total, reduction="sum").item()   # type: ignore

                epoch_val_loss = epoch_val_loss / epoch_val_samples
                epoch_val_accuracy = 100.0 * epoch_val_correct / epoch_val_samples

        if scheduler is not None:
            scheduler.step()

        if accelerator.is_main_process:
            epoch_time = time.perf_counter() - start_time
            # Print both loss and accuracy at the end of each epoch
            temp_message = f"Epoch {epoch+1}/{epochs} Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%,"
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
            history["lr"].append(optimizer.param_groups[0]['lr'])
            print(temp_message)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if save_location is not None:
            if save_model:
                model_name = f"{save_prefix}_model.pth" if save_prefix else "model.pth"
                model_savepath = save_location / model_name
                model_savepath.parent.mkdir(parents=True, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model, keep_torch_compile=False)
                torch.save(unwrapped_model.state_dict(), model_savepath)

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
