import os

import torch
from accelerate import Accelerator
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)


def profile_training_loop(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None = None,
    augmentations: torch.nn.Module | None = None,
    val_augmentations: torch.nn.Module | None = None,
    epochs: int = 1,
    log_dir: str = "./profile_logs",
):
    """A simplified training function designed around profiling CPU and GPU usage. Use to optimize batch size and augmentations.

    Args:
        accelerator (Accelerator): The HuggingFace Accelerator instance managing device placement and precision.
        model (torch.nn.Module): The model that will be benchmarked.
        optimizer (torch.optim.Optimizer): The optimizer that will be used during profiling.
        criterion (torch.nn.Module): The loss criterion that will be benchmarked.
        train_loader (torch.utils.data.DataLoader): The train DataLoader used for benchmarking.
        val_loader (torch.utils.data.DataLoader, optional): The validation DataLoader. Defaults to None.
        augmentations (torch.nn.Module, optional): On-device augmentations applied to training batches. Defaults to None.
        val_augmentations (torch.nn.Module, optional): On-device augmentations applied to validation batches. Defaults to None.
        epochs (int, optional): The training epochs. Profiling takes place according to the schedule. Defaults to 1.
        log_dir (str, optional): Directory where TensorBoard trace files and per-rank profiling traces are stored. Defaults to "./profile_logs".
    """

    batches_per_epoch = len(train_loader)
    my_schedule = schedule(
        skip_first=batches_per_epoch
        + 5,  # Skip the first epoch and 5 batches to avoid warm-up overhead
        wait=2,  # Wait 2 batches before starting warm-up
        warmup=2,  # Warm up profiler for 2 batches
        active=4,  # Record traces for 4 active batches
        repeat=1,  # Run the cycle once
    )

    # Save per-rank traces
    rank_trace_dir = os.path.join(log_dir, f"rank_{accelerator.process_index}")
    trace_handler = tensorboard_trace_handler(rank_trace_dir)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,  # Optional: trace VRAM usage
        with_stack=False,  # True adds substantial overhead
    ) as prof:
        for _ in range(epochs):
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

                optimizer.zero_grad(set_to_none=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                accelerator.backward(loss)
                optimizer.step()
                prof.step()

            # Validation Phase (unprofiled evaluation, included for a more realistic scenario)
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, labels = batch
                        inputs = inputs.to(memory_format=torch.channels_last)
                        if val_augmentations is not None:
                            inputs, labels = val_augmentations(inputs, labels)
                        if labels.ndim > 1:
                            labels = labels.argmax(dim=-1).squeeze().long()

                        outputs = model(inputs)
                        if isinstance(outputs, (list, tuple)):
                            outputs = outputs[0]
                        _ = criterion(outputs, labels)

    accelerator.wait_for_everyone()

    # Print summary table from rank 0 only
    if accelerator.is_main_process:
        accelerator.print(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
        )


def profile_distillation_loop():
    pass
