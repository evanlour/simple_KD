from simple_kd.generators.synthetic_noise import dead_leaves, generate_feature_samples, generate_feature_combinations
import os
import cv2
import torch
import torchvision.models as models
from simple_kd.utils import apply_hooks

if __name__ == "__main__":
    os.makedirs("./tests/images", exist_ok=True)
    dead_leaves_img1 = dead_leaves(256, 3)
    dead_leaves_img2 = dead_leaves(128, 1, rmin=0.03)

    cv2.imwrite("./tests/images/dead_leaves_img1.png", cv2.cvtColor(dead_leaves_img1.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("./tests/images/dead_leaves_img2.png", dead_leaves_img2[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    target_layer = ['avgpool']
    features = apply_hooks(model=model, layers=target_layer, detach=False)

    combos = generate_feature_combinations(samples=2, embedding_size=512, max_order=1)
    gen_imgs = generate_feature_samples(model=model, features=features,targets=combos, device=device) # type: ignore

    for i, img in enumerate(gen_imgs):
        filename = os.path.join("./tests/images", f"sample_{i:02d}_feature_vis.png")
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
