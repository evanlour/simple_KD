import torch
import torchvision.models as models
from simple_kd.utils import apply_hooks
from simple_kd.generators.synthetic_noise import generate_feature_dataset, generate_dead_leaves_dataset
from pathlib import Path

DEAD_LEAVES_TARGET_LOCATION = "/mnt/storage/Datasets/dead_leaves"
FEATURE_VIS_TARGET_LOCATION = "/mnt/storage/Datasets/feature_vis"
IMAGE_SIZE=32
NUM_IMAGES = 50000

if __name__ == "__main__":
    # Dead Leaves Dataset
    generate_dead_leaves_dataset(res=IMAGE_SIZE, channels=3, savepath=Path(DEAD_LEAVES_TARGET_LOCATION), parallel_workers=3, samples=NUM_IMAGES)

    # Feature vis Dataset (and preparation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    target_layer = ['avgpool']
    features = apply_hooks(model=model, layers=target_layer, detach=False)

    generate_feature_dataset(model=model,
    features=features,
    num_imgs=NUM_IMAGES,
    batch_size=1024,
    device=device, # type: ignore
    savepath=Path(FEATURE_VIS_TARGET_LOCATION),
    embedding_size=512,
    img_size=IMAGE_SIZE,
    mean=torch.tensor([0.5072, 0.4865, 0.4406], device=device).view(1, 3, 1, 1),
    std=torch.tensor([0.2673, 0.2564, 0.2761], device=device).view(1, 3, 1, 1))
