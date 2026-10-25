import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import Subset
from torchvision.models import resnet18, resnet34
from torchvision.transforms import v2

from simple_kd.losses import DKDLoss
from simple_kd.training import distill, evaluate_model
from simple_kd.utils import Dataset

CLASSES = 100
EPOCHS = 100
BATCH = 256
WORKERS = 2

TRAIN_IMAGES_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_train_images.npy"
TRAIN_LABELS_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_train_labels.npy"
VAL_IMAGES_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_val_images.npy"
VAL_LABELS_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_val_labels.npy"
TEST_IMAGES_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_test_images.npy"
TEST_LABELS_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_test_labels.npy"
TEACHER_MODEL_LOCATION = (
    "/mnt/storage/Datasets_results/CIFAR100_train_results/model.pth"
)
TARGET_SAVE_LOCATION = "/mnt/storage/Datasets_results/CIFAR100_distillation_results"

CPU_AUGS = v2.Compose([v2.ToImage()])

GPU_AUGS = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5072, 0.4865, 0.4406], std=[0.2673, 0.2564, 0.2761]),
        v2.RandomErasing(p=0.35, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        v2.RandomChoice(
            [
                v2.CutMix(num_classes=CLASSES, alpha=1.0),
                v2.MixUp(num_classes=CLASSES, alpha=0.8),
                v2.Identity(),
            ],
            p=[0.4, 0.4, 0.2],
        ),
    ]
)

VAL_AUGS = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5072, 0.4865, 0.4406], std=[0.2673, 0.2564, 0.2761]),
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

    # Load Dataset
    full_trainset = Dataset(
        TRAIN_IMAGES_LOCATION,
        TRAIN_LABELS_LOCATION,
        loading_mode="r+",
        augmentations=CPU_AUGS,
    )
    valset = Dataset(
        VAL_IMAGES_LOCATION,
        VAL_LABELS_LOCATION,
        loading_mode="r+",
        augmentations=CPU_AUGS,
    )
    # We will simulate a low data run by dumping 80% of the CIFAR100 data
    full_labels = np.load(
        TRAIN_LABELS_LOCATION
    )  # Labels are small so it is okay to load them all
    indices = np.arange(len(full_labels))

    # We will keep only the 20% of the indices.
    subset_idx, _ = train_test_split(
        indices,
        train_size=0.2,
        stratify=full_labels,
        random_state=42,
    )
    subset_idx = np.sort(
        subset_idx
    )  # Supposedly better read times, will need to be tested
    trainset = Subset(full_trainset, subset_idx)  # type: ignore

    generator = torch.Generator()

    generator.manual_seed(58709)
    train_loader = data.DataLoader(
        trainset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        generator=generator,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )

    generator.manual_seed(58709)
    val_loader = data.DataLoader(
        valset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        generator=generator,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )

    teacher_model = resnet34()
    teacher_model.conv1 = torch.nn.Conv2d(
        in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
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
        in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
    )
    student_model.maxpool = torch.nn.Identity()  # type: ignore
    student_feats = student_model.fc.in_features
    student_model.fc = torch.nn.Linear(feats, CLASSES)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    distillation_criterion = DKDLoss()
    student_optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        fused=True,
    )
    student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        student_optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    student_model = student_model.to(memory_format=torch.channels_last)  # type: ignore
    teacher_model = teacher_model.to(memory_format=torch.channels_last)  # type: ignore
    student_model, teacher_model, train_loader, val_loader, student_optimizer = (
        accelerator.prepare(
            student_model, teacher_model, train_loader, val_loader, student_optimizer
        )
    )
    student_model = torch.compile(student_model)
    teacher_model = torch.compile(teacher_model)

    GPU_AUGS = GPU_AUGS.to(device)
    VAL_AUGS = VAL_AUGS.to(device)
    criterion = criterion.to(device)  # prepare only when there are trainable parameters
    distillation_criterion = distillation_criterion.to(device)

    history = distill(
        accelerator=accelerator,
        teacher_model=teacher_model,  # type: ignore
        student_model=student_model,  # type: ignore
        train_loader=train_loader,
        student_optimizer=student_optimizer,
        criterion=criterion,
        distillation_criterion=distillation_criterion,
        epochs=EPOCHS,
        augmentations=GPU_AUGS,
        val_augmentations=VAL_AUGS,
        student_scheduler=student_scheduler,
        val_loader=val_loader,
        save_location=Path(TARGET_SAVE_LOCATION),
        save_history=True,
        save_student_model=True,
        save_teacher_model=False,
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
