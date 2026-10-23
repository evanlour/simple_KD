import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34
from torchvision.transforms import v2

from simple_kd.data_free import odsd_sample, odsd_train
from simple_kd.training import evaluate_model
from simple_kd.utils import LabelFreeDataset

CLASSES = 100
EPOCHS = 5
BATCH = 256
WORKERS = 2
KMEANS_SIZE = 7
KEEP_RATIO = 0.5
SKIP_SAMPLING = False

EXTERNAL_IMAGES_LOCATION = "/mnt/storage/Datasets/stylegan/stylegan.npy"
VAL_IMAGES_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_val_images.npy"
VAL_LABELS_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_val_labels.npy"
TEST_IMAGES_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_test_images.npy"
TEST_LABELS_LOCATION = "/mnt/storage/Datasets/CIFAR100/CIFAR100_test_labels.npy"
TEACHER_MODEL_LOCATION = "/mnt/storage/Datasets_results/CIFAR100_test_results/model.pth"
TARGET_SAVE_LOCATION = "/mnt/storage/Datasets_results/CIFAR100_odsd_results"
SAMPLED_IMAGES_PATH = Path(TARGET_SAVE_LOCATION) / "sampled_images.npy"

CPU_AUGS = v2.Compose([v2.ToImage()])

GPU_HIGH_AUGS = v2.Compose([
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5072, 0.4865, 0.4406], std=[0.2673, 0.2564, 0.2761]),
])

GPU_LOW_AUGS = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5072, 0.4865, 0.4406], std=[0.2673, 0.2564, 0.2761]),
])

TEST_AUGS = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5072, 0.4865, 0.4406], std=[0.2673, 0.2564, 0.2761]),
])


def seed_worker(worker_id):
    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    accelerator = Accelerator(
        mixed_precision="bf16",
        split_batches=True,
        step_scheduler_with_optimizer=False,
    )
    device = accelerator.device

    torch.manual_seed(42)
    np.random.seed(22)
    random.seed(42)
    torch.backends.cudnn.benchmark = True # Enable for performance
    # torch.use_deterministic_algorithms(True, warn_only=True) # Comment for performance
    torch.set_float32_matmul_precision("high") # From least to most performant: ["Highest", "High", "Medium"]
    torch.backends.cuda.matmul.allow_tf32 = True # Disable if unsupported
    torch.backends.cudnn.allow_tf32 = True # Disable if unsupported

    external_dataset = LabelFreeDataset(images_path=EXTERNAL_IMAGES_LOCATION, augmentations=CPU_AUGS, loading_mode="r+")

    generator = torch.Generator()
    generator.manual_seed(58709)

    external_loader = DataLoader(
        external_dataset,
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
        in_channels=3,
        out_channels=64,
        kernel_size=5,
        stride=1,
        padding=2,
        bias=False,
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
        in_channels=3,
        out_channels=64,
        kernel_size=5,
        stride=1,
        padding=2,
        bias=False
    )
    student_model.maxpool = torch.nn.Identity() # type: ignore
    student_feats = student_model.fc.in_features
    student_model.fc = torch.nn.Linear(feats, CLASSES)

    student_optimizer = torch.optim.SGD(student_model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9, nesterov=False, fused=True)
    student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=EPOCHS, eta_min=2e-3)

    student_model = student_model.to(memory_format=torch.channels_last) #type: ignore
    teacher_model = teacher_model.to(memory_format=torch.channels_last) #type: ignore

    student_model, teacher_model, student_optimizer, external_loader = accelerator.prepare(student_model, teacher_model, student_optimizer, external_loader)

#     student_model = torch.compile(student_model)
#     teacher_model = torch.compile(teacher_model)


    GPU_LOW_AUGS = GPU_LOW_AUGS.to(device)
    GPU_HIGH_AUGS = GPU_HIGH_AUGS.to(device)

    if not SKIP_SAMPLING:
        odsd_sample(
            teacher_model=teacher_model,
            dataloader=external_loader,
            augmentations=GPU_LOW_AUGS,
            num_classes=CLASSES,
            save_location=Path(TARGET_SAVE_LOCATION),
            save_plots=True
        )

    sampled_dataset_path = Path(TARGET_SAVE_LOCATION) / "sampled_dataset.npy"
    sampled_dataset = LabelFreeDataset(images_path=str(sampled_dataset_path), augmentations=CPU_AUGS, loading_mode="r+")

    generator.manual_seed(58709)
    sampled_loader = DataLoader(
        sampled_dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        generator=generator,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )
    sampled_loader = accelerator.prepare(sampled_loader)

    validation_params = {
        "validation_images": Path(VAL_IMAGES_LOCATION),
        "validation_labels": Path(VAL_LABELS_LOCATION),
        "validation_augs": TEST_AUGS,
        "classes": CLASSES,
        "batch_size": BATCH,
    }

    history = odsd_train(
        accelerator=accelerator,
        student_model=student_model,  # type: ignore
        teacher_model=teacher_model,  # type: ignore
        train_loader=sampled_loader,
        student_optimizer=student_optimizer,
        epochs=EPOCHS,
        augmentations=GPU_HIGH_AUGS,
        basic_augmentations=GPU_LOW_AUGS,
        scheduler=student_scheduler,
        save_location=Path(TARGET_SAVE_LOCATION),
        save_history=True,
        save_student_model=True,
        validation_metrics=validation_params,
    )

    print("Student model evaluation:")
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
