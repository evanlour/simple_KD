import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torchvision.models import resnet18, resnet34
from torchvision.transforms import v2

from simple_kd.data_free import zskt_train
from simple_kd.generators import SimpleGenerator
from simple_kd.training import evaluate_model
from simple_kd.utils.feature_selector import apply_hooks

CLASSES = 100
EPOCHS = 500
BATCH = 512
WORKERS = 2
GENERATOR_STEPS_PER_EPOCH = 1
MODEL_STEPS_PER_EPOCH = 50
GENERATOR_LATENT_SIZE = 128

VAL_IMAGES_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_val_images.npy"
VAL_LABELS_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_val_labels.npy"
TEST_IMAGES_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_test_images.npy"
TEST_LABELS_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_test_labels.npy"
TEACHER_MODEL_LOCATION = "/mnt/storage/Datasets_results/CIFAR100_test_results/model.pth"
TARGET_SAVE_LOCATION = "/mnt/storage/Datasets_results/CIFAR100_zskt_results"

CPU_AUGS = v2.Compose([v2.ToImage()])

GPU_AUGS = v2.Compose(
    [
        # v2.RandomHorizontalFlip(p=0.5),
        # v2.RandAugment(num_ops=2, magnitude=9), # RandAugment is unsafe and crashes the generator!
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5072, 0.4865, 0.4406], std=[0.2673, 0.2564, 0.2761]),
        # v2.RandomErasing(p=0.35, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
    ]
)

TEST_AUGS = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5072, 0.4865, 0.4406], std=[0.2673, 0.2564, 0.2761]),
    ]
)


# A helpful function, preventing workers from hogging multiple threads and ensuring reproducibility if needed
def seed_worker(worker_id):
    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    accelerator = Accelerator(
        mixed_precision="bf16",
        split_batches=True,  # For multi GPU, the batch is global instead of per GPU when true
        step_scheduler_with_optimizer=False,  # Set to false if scheduler is updated per epoch
    )
    device = accelerator.device
    torch.autograd.set_detect_anomaly(True)
    # Set deterministic algorithms for reproducibility
    torch.manual_seed(42)
    np.random.seed(22)
    random.seed(42)
    torch.backends.cudnn.benchmark = True  # Enable for performance
    # torch.use_deterministic_algorithms(True, warn_only=True) # Comment for performance
    torch.set_float32_matmul_precision(
        "high"
    )  # From least to most performant: ["Highest", "High", "Medium"]
    torch.backends.cuda.matmul.allow_tf32 = True  # Disable if unsupported
    torch.backends.cudnn.allow_tf32 = True  # Disable if unsupported

    teacher_model = resnet34()
    teacher_model.conv1 = torch.nn.Conv2d(
        in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False
    )
    teacher_model.maxpool = torch.nn.Identity()  # type: ignore
    feats = teacher_model.fc.in_features
    teacher_model.fc = torch.nn.Linear(feats, CLASSES)
    checkpoint = torch.load(TEACHER_MODEL_LOCATION)
    teacher_model.load_state_dict(checkpoint)
    # To make sure the teacher is frozen
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model = resnet18()
    student_model.conv1 = torch.nn.Conv2d(
        in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False
    )
    student_model.maxpool = torch.nn.Identity()  # type: ignore
    student_feats = student_model.fc.in_features
    student_model.fc = torch.nn.Linear(feats, CLASSES)

    # Hook intermediate layers for both models
    teacher_stages = apply_hooks(
        teacher_model, ["layer1.2.bn2", "layer2.3.bn2", "layer3.2.bn2", "layer4.2.bn2"]
    )
    student_stages = apply_hooks(
        student_model, ["layer1.1.bn2", "layer2.1.bn2", "layer3.1.bn2", "layer4.1.bn2"]
    )

    image_generator = SimpleGenerator(32, 128, 3)
    student_optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=1e-1,
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=False,
        fused=True,
    )
    student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        student_optimizer, T_max=EPOCHS, eta_min=2e-3
    )
    generator_optimizer = torch.optim.AdamW(
        image_generator.parameters(),
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        fused=True,
    )
    generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        generator_optimizer, T_max=EPOCHS, eta_min=5e-4
    )

    student_model = student_model.to(memory_format=torch.channels_last)  # type: ignore
    teacher_model = teacher_model.to(memory_format=torch.channels_last)  # type: ignore
    image_generator = image_generator.to(memory_format=torch.channels_last)  # type: ignore
    (
        student_model,
        teacher_model,
        student_optimizer,
        image_generator,
        generator_optimizer,
    ) = accelerator.prepare(
        student_model,
        teacher_model,
        student_optimizer,
        image_generator,
        generator_optimizer,
    )
    student_model = torch.compile(student_model)
    teacher_model = torch.compile(teacher_model)
    image_generator = torch.compile(image_generator)

    GPU_AUGS = GPU_AUGS.to(device)

    validation_params = {
        "validation_images": Path(VAL_IMAGES_LOCATION),
        "validation_labels": Path(VAL_LABELS_LOCATION),
        "validation_augs": TEST_AUGS,
        "classes": CLASSES,
    }

    _ = zskt_train(
        accelerator=accelerator,
        student_model=student_model,  # type: ignore
        teacher_model=teacher_model,  # type: ignore
        generator=image_generator,  # type: ignore
        augmentations=GPU_AUGS,
        batch_size=BATCH,
        student_optimizer=student_optimizer,
        generator_optimizer=generator_optimizer,
        epochs=EPOCHS,
        generator_latent_size=GENERATOR_LATENT_SIZE,
        generator_steps=GENERATOR_STEPS_PER_EPOCH,
        model_steps=MODEL_STEPS_PER_EPOCH,
        temperature=4.0,
        model_at_gamma=0.0,
        generator_at_gamma=0.0,
        scheduler=student_scheduler,
        generator_scheduler=generator_scheduler,
        student_stages=student_stages,
        teacher_stages=teacher_stages,
        save_location=Path(TARGET_SAVE_LOCATION),
        save_history=True,
        save_student_model=True,
        save_generator=True,
        validation_metrics=validation_params,
    )

    print("Student model results")
    _ = evaluate_model(
        model=student_model,  # type: ignore
        images=Path(TEST_IMAGES_LOCATION),
        labels=Path(TEST_LABELS_LOCATION),
        batch_size=BATCH,
        device=device,
        augs=TEST_AUGS,
        criterion=torch.nn.CrossEntropyLoss(),
        classes=CLASSES,
        save_folder=Path(TARGET_SAVE_LOCATION),
        save_prefix="student",
    )

    print("Teacher model results")
    _ = evaluate_model(
        model=teacher_model,  # type: ignore
        images=Path(TEST_IMAGES_LOCATION),
        labels=Path(TEST_LABELS_LOCATION),
        batch_size=BATCH,
        device=device,
        augs=TEST_AUGS,
        criterion=torch.nn.CrossEntropyLoss(),
        classes=CLASSES,
        save_folder=None,
    )
