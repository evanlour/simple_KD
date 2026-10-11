from accelerate import Accelerator
import os
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

def profile_training_loop(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None=None,
    augmentations: torch.nn.Module | None=None,
    val_augmentations: torch.nn.Module | None=None,
    epochs:int =1,
    log_dir:str ="./profile_logs"
):
    """ A simplified train function designed around profiling cpu and gpu usage. Use to optimize batch size and augmentations.

    Args:
        model (torch.nn.Module): The model that will be benchmarked.
        train_loader (torch.utils.data.DataLoader): The train data that will be used for benchmarking.
        optimizer (torch.optim): The optimizer that will be used.
        device (torch.device): The device where the benchmarking will take place.
        criterion (torch.nn.Module): The loss criterion that will be benchmarked.
        epochs (int, optional): The training epochs, no need for a lot of them, as the benchmarking takes place on the second epoch. Defaults to 2.
        augmentations (torch.nn.Module, optional): The augmentations that will be applied to the train data. Defaults to None.
        val_augmentations (torch.nn.Module, optional): The augmentations that will be applied to the val data. Defaults to None.
        val_loader (torch.utils.data.DataLoader, optional): The validation data that will be used. Defaults to None.
        amp (torch.dtype, optional): Automatic mixed precision. The accepted dtypes depend on device. Usually torch.float32, torch.float16, torch.bfloat16. Defaults to torch.float32.
    """

    batches_per_epoch = len(train_loader)
    my_schedule = schedule(
        skip_first=batches_per_epoch + 5, # Skip the first epoch and 5 more batches
        wait=2,                           # Check per 2 batches (or prof.step())
        warmup=2,                         # Warmup
        active=4,                         # Record 4 batches for data
        repeat=1                          # Run only once
    )

    # Save per-rank traces
    rank_trace_dir = os.path.join(log_dir, f"rank_{accelerator.process_index}")
    trace_handler = tensorboard_trace_handler(rank_trace_dir)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True, # Optional: trace VRAM usage
        with_stack=False     # True adds substantial overhead
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
        accelerator.print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

def profile_distillation_loop():
    pass
