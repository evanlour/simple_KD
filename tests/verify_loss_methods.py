import inspect

import torch
from accelerate import Accelerator
from torch.utils import benchmark

from simple_kd.losses import SkepticalClassifier
from simple_kd.losses.functional import (
    at_loss,
    dkd_loss,
    kd_loss,
    pkt_loss,
    rkd_angle_loss,
    rkd_dist_loss,
    skeptical_loss,
    sp_loss,
    ssp_loss,
)

KD_FUNCTIONS = {
    "KD": kd_loss,
    "DKD": dkd_loss,
    "PKT": pkt_loss,
    "SSP": ssp_loss,
    "AT": at_loss,
    "RKDAngle": rkd_angle_loss,
    "RKDDist": rkd_dist_loss,
    "SP": sp_loss,
    "Skeptical": skeptical_loss,
}

KD_CONFIGS = {
    "KD": {
        "T": 4.0,
        "gamma": 1.0,
    },
    "DKD": {"T": 4.0, "tckd_gamma": 1.0, "nckd_gamma": 8.0},
    "PKT": {
        "stage_percentages": (0.3, 0.3, 0.2, 0.2),
        "gamma": 128.0,
    },
    "SSP": {
        "stage_percentages": (0.3, 0.3, 0.2, 0.2),
        "gamma": 128.0,
        "sort_channels": True,
    },
    "AT": {"stage_percentages": (0.3, 0.3, 0.2, 0.2), "gamma": 128.0, "pow": 2.0},
    "RKDAngle": {
        "stage_percentages": (0.3, 0.3, 0.2, 0.2),
        "gamma": 128.0,
    },
    "RKDDist": {
        "stage_percentages": (0.3, 0.3, 0.2, 0.2),
        "gamma": 128.0,
    },
    "SP": {
        "stage_percentages": (0.3, 0.3, 0.2, 0.2),
        "gamma": 128.0,
    },
    "Skeptical": {"T": 4.0, "kd_gamma": 1.0, "in_kd_gamma": 1.0},
}


def filter_function_call(fn, **kwargs):
    """Inner working function for the benchmark test file, removes non relevant data provided to the function.

    Args:
        fn (function): The function that will be cleared.
        **kwargs: All other relevant or not parameters.

    Returns:
        The functions, original return item.
    """
    params = inspect.signature(fn).parameters
    return fn(**{k: v for k, v in kwargs.items() if k in params})


def supported_dtypes(device: torch.device) -> list[torch.dtype]:
    """Checks what type of torch.dtype float options are available on the device.

    Args:
        device (torch.device): The device that will be checked.

    Returns:
        List[torch.dtype]: The list of available float torch.Tensor dtypes.
    """
    supported = [torch.float32]

    try:
        x = torch.randn(2, 2, dtype=torch.float16, device=device)
        _ = (x @ x).sum()
        supported.append(torch.float16)
    except ValueError:
        print("dtype=float16 is not supported on current device")

    try:
        x = torch.randn(2, 2, dtype=torch.bfloat16, device=device)
        _ = (x @ x).sum()
        supported.append(torch.bfloat16)
    except ValueError:
        print("dtype=bfloat16 is not supported on current device")

    return supported


def cast_batch_tensors(batch: dict, dtype: torch.dtype) -> dict:
    """Internal function that converts a dictionary of tensors to the given torch.dtype. Does not convert ints.

    Args:
        batch (dict): Dictionary of tensor values.
        dtype (torch.dtype): The dtype target.

    Returns:
        Dict: The dictionary with the float tensor values changed to the given dtype.
    """
    cast_dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            cast_dict[k] = v.detach().to(dtype=dtype).requires_grad_(v.requires_grad)
        elif isinstance(v, list):
            cast_dict[k] = [
                elem.detach().to(dtype=dtype).requires_grad_(elem.requires_grad)
                if isinstance(elem, torch.Tensor) and elem.is_floating_point()
                else elem
                for elem in v
            ]
        elif isinstance(v, torch.nn.Module):
            cast_dict[k] = v.to(dtype=dtype)
        else:
            cast_dict[k] = v
    return cast_dict


def measure_peak_vram_mb(
    fn, batch_tensors: dict, config: dict, device: torch.device
) -> float:
    """Measures transient VRAM allocated during forward + backward (in MB).

    Args:
        fn (function): The function that will be tested.
        batch_tensors (dict): Dictionary containing the input tensors the method will use.
        config (dict): Dictionary containing the parameters the method will use.
        device (torch.device): The device where the function will run.

    Returns:
        float: The peak ram usage of the given method.
    """
    if device.type != "cuda":
        return 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device)

    # Run forward + backward pass
    loss = filter_function_call(fn, **batch_tensors, **config)
    loss.backward(retain_graph=True)

    peak_mem = torch.cuda.max_memory_allocated(device)
    mem_overhead_mb = max(0.0, (peak_mem - base_mem) / (1024**2))  # MB
    return mem_overhead_mb


def benchmark_method(
    kd_functions: dict, kd_configs: dict, input_tensors: dict, accelerator: Accelerator
):
    """Benchmarks a dictionary of functions with their given configuration on a given device with given tensors.

    Args:
        kd_functions (Dict): Dictionary containing the functions that will be used.
        kd_configs (Dict): Dictionary containing the function parameters that will be used.
        input_tensors (Dict): The input tensors that will be used. From model outputs to labels and images.
        accelerator (Accelerator): The accelerator device(s) where the benchmark will run.
    """
    device = accelerator.device
    types = supported_dtypes(device)
    dtype_names = [str(d).split(".")[-1] for d in types]
    time_data = {name: {} for name in kd_functions}
    ram_data = {name: {} for name in kd_functions}

    for dtype in types:
        current_dtype = str(dtype).split(".")[-1]
        current_batch = cast_batch_tensors(input_tensors, dtype)

        for name, kd_func in kd_functions.items():
            method_config = kd_configs[name]
            timer = benchmark.Timer(
                stmt="loss = call(kd_func, **current_batch, **config); loss.backward(retain_graph=True)",
                globals={
                    "call": filter_function_call,
                    "kd_func": kd_func,
                    "current_batch": current_batch,
                    "config": method_config,
                },
                sub_label=f"{name:<6}",
                label=current_dtype,
                description=f"Distillation benchmark for {device.type.upper()}",
                num_threads=1,
            )
            # Specific device benchmarks
            measurement = timer.blocked_autorange(min_run_time=0.5)
            local_time = torch.tensor(measurement.median * 1e6, device=device)
            local_mem = torch.tensor(
                measure_peak_vram_mb(kd_func, current_batch, method_config, device),
                device=device,
            )

            # Average across all accelerators for reporting
            reduced_time = accelerator.reduce(local_time, reduction="mean")
            reduced_mem = accelerator.reduce(local_mem, reduction="mean")
            assert isinstance(reduced_time, torch.Tensor)
            assert isinstance(reduced_mem, torch.Tensor)
            avg_time = reduced_time.item()
            avg_mem = reduced_mem.item()

            time_data[name][current_dtype] = avg_time
            ram_data[name][current_dtype] = avg_mem

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        col_width = 24
        header_cols = "".join([f"| {col:^{col_width}} " for col in dtype_names])
        divider = "-" * (14 + len(dtype_names) * (col_width + 3))

        accelerator.print(
            f"\n[-- Distillation Benchmark: Speed & Peak VRAM for {accelerator.num_processes} process(es) and ({device.type.upper()}) --]"
        )
        accelerator.print(f"{'Method':<12} {header_cols}")
        accelerator.print(divider)

        for name in kd_functions:
            row_cells = []
            for d in dtype_names:
                t = time_data[name].get(d, 0.0)
                m = ram_data[name].get(d, 0.0)
                # Display format: 405.66 µs / 1.24 MB (or N/A if on CPU)
                mem_str = f"{m:6.2f} MB" if device.type == "cuda" else "N/A"
                cell = f"{t:>8.2f} µs / {mem_str}"
                row_cells.append(f"| {cell:<{col_width}} ")

            accelerator.print(f"{name:<12} " + "".join(row_cells))

        accelerator.print(divider)


if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(
        f"Started testing with {accelerator.num_processes} processes and {device} device"
    )
    torch.manual_seed(42 + accelerator.process_index)
    dummy_student = torch.randn(64, 64, requires_grad=True, device=device)
    dummy_teacher = torch.randn(64, 64, requires_grad=True, device=device)
    dummy_inter_student = [
        torch.randn(64, 64, 32, 32, requires_grad=True, device=device)
    ]
    dummy_inter_teacher = [
        torch.randn(64, 128, 16, 16, requires_grad=True, device=device)
    ]
    dummy_images = torch.randn(64, 3, 128, 128, device=device)
    dummy_labels = torch.randint(low=0, high=63, size=(64,), device=device)
    dummy_classifier = SkepticalClassifier(64, 64).to(device=device)
    batch_tensor = {
        "student_output": dummy_student,
        "teacher_output": dummy_teacher,
        "student_stages": dummy_inter_student,
        "teacher_stages": dummy_inter_teacher,
        "labels": dummy_labels,
        "images": dummy_images,
        "aux_classifier": dummy_classifier,
    }

    # First validate that the methods work properly
    for name, kd_func in KD_FUNCTIONS.items():
        method_config = KD_CONFIGS[name]
        loss = filter_function_call(kd_func, **batch_tensor, **method_config)
        assert loss.dim() == 0, (
            f"{name} returned item of shape {loss.shape} instead of number"
        )
        assert torch.isfinite(loss), f"{name}: loss is NaN or Inf ({loss.item()})"
        assert loss >= 0.0, f"{name}: loss is negative ({loss.item()})"
        loss.backward()
        assert dummy_student.grad is not None and dummy_student.grad.abs().sum() > 0, (
            f"{name}: no non-zero gradient flow to student"
        )

    # Zero gradients before benchmarking
    dummy_student.grad = None
    if hasattr(dummy_classifier, "zero_grad"):
        dummy_classifier.zero_grad()

    accelerator.wait_for_everyone()
    # Benchmark the methods
    benchmark_method(KD_FUNCTIONS, KD_CONFIGS, batch_tensor, accelerator)

    print("If you see this, methods at first glance work...")
