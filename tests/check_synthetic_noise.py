from simple_kd.generators.synthetic_noise import dead_leaves, generate_feature_samples, generate_feature_combinations, generate_statistical_sample
import os
import cv2
import torch
import torchvision.models as models
from simple_kd.generators.synthetic_noise.stylegan import generate_stylegan_samples
from simple_kd.utils import apply_hooks
from torchvision.utils import save_image

if __name__ == "__main__":
    os.makedirs("./tests/images", exist_ok=True)
    # Dead Leaves
    dead_leaves_img1 = dead_leaves(256, 3)
    dead_leaves_img2 = dead_leaves(128, 1, rmin=0.03)

    cv2.imwrite("./tests/images/dead_leaves_sample1.png", cv2.cvtColor(dead_leaves_img1.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
    cv2.imwrite("./tests/images/dead_leaves_sample2.png", dead_leaves_img2[0])

    # Feature visualization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    target_layer = ['avgpool']
    features = apply_hooks(model=model, layers=target_layer, detach=False)

    combos = generate_feature_combinations(samples=2, embedding_size=512, max_order=1)
    gen_imgs = generate_feature_samples(model=model, features=features,targets=combos, device=device) # type: ignore

    for i, img in enumerate(gen_imgs):
        filename = os.path.join("./tests/images", f"feature_vis_sample_{i:02d}.png")
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Statistical images
    with torch.no_grad():
        statistical_samples = generate_statistical_sample(B=8, resolution=128, n_iter=10, device=torch.device(device))
        save_image(statistical_samples.float() / 255.0, "./tests/images/statistical_samples.png", nrow=4)

    # StyleGAN images
    for var in ['random', 'highfreq', 'sparse', 'oriented']:
        stylegan_imgs = generate_stylegan_samples(device=device, variant=var, resolution=128, n_samples=2) # type: ignore
        for i, img in enumerate(stylegan_imgs):
            filename = os.path.join("./tests/images", f"stylegan_{var}_sample_{i:02d}.png")
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
