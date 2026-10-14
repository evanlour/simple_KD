from itertools import combinations
import numpy as np
import torch
from pathlib import Path

class ColorDecorrelator:
    COLOR_CORR = torch.tensor([
        [0.56,  0.22, -0.04],
        [0.58,  0.72, -0.10],
        [0.58, -0.65,  0.99]
    ], dtype=torch.float32)

    def __init__(self, device: torch.device):
        mat = self.COLOR_CORR / torch.linalg.norm(self.COLOR_CORR, dim=0, keepdim=True)
        self.mat = mat.to(device)

    def to_rgb(self, latent_spatial: torch.Tensor) -> torch.Tensor:
        """Maps latent space -> RGB in [0, 1] using smooth sigmoid projection."""
        rgb = torch.einsum("bchw,cd->bdhw", latent_spatial, self.mat)
        return torch.sigmoid(rgb)

class FFTImage(torch.nn.Module):
	def __init__(self, h: int, w: int, batch_size: int, device: torch.device):
		super().__init__()
		self.h, self.w = h, w
		self.device = device

		# 1/f Frequency grid
		fy = torch.fft.fftfreq(h)[:, None]
		fx = torch.fft.rfftfreq(w)[None, :]
		freq_radial = torch.sqrt(fy**2 + fx**2)
		scale = 1.0 / torch.maximum(freq_radial, torch.tensor(1.0 / max(h, w)))
		self.scale = scale.unsqueeze(0).unsqueeze(0).to(self.device)

		# Trainable Fourier spectrum
		self.spectrum = torch.nn.Parameter(
			torch.randn(batch_size, 3, h, w // 2 + 1, dtype=torch.cfloat, device=self.device) * 0.01
		)

	def forward(self) -> torch.Tensor:
		"""Returns spatial latent tensor via inverse FFT."""
		scaled_spectrum = self.spectrum * self.scale
		return torch.fft.irfft2(scaled_spectrum, s=(self.h, self.w), norm="ortho")

def generate_feature_combinations(samples: int, embedding_size: int, max_order: int = 3) -> list[list[int]]:
    """Generates neuron combinations from 1 to max_order."""
    combos = []
    for r in range(1, max_order + 1):
        for c in combinations(range(embedding_size), r):
            combos.append(list(c))
            if len(combos) == samples:
                return combos
    return combos

def generate_feature_samples(
    model: torch.nn.Module,
    features: list[torch.Tensor],
    targets: list[list[int]],
    device: torch.device,
    img_size: int = 224,
    n_iters: int = 120,
    lr: float = 0.05,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None) -> np.ndarray:

	if mean is None:
		mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
	if std is None:
		std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
	batch_size = len(targets)

	fft_img = FFTImage(h=img_size, w=img_size, batch_size=batch_size, device=device)
	decorrelator = ColorDecorrelator(device)
	optimizer = torch.optim.AdamW(fft_img.parameters(), lr=lr)

	for _ in range(n_iters):
		optimizer.zero_grad()

		spatial = fft_img()
		rgb = decorrelator.to_rgb(spatial)

		# Light random jitter (pad 8px and random crop) to prevent grid artifacts
		pad = 8
		padded = torch.nn.functional.pad(rgb, [pad] * 4, mode="reflect")
		dy, dx = torch.randint(0, 2 * pad, (2,))
		jittered = padded[:, :, dy : dy + img_size, dx : dx + img_size]

		normed = (jittered - mean) / std
		_ = model(normed)

		act = features[0].mean(dim=(-2, -1)) # Spatial average pooling: (B, 512, 7, 7) -> (B, 512)

		loss = -torch.stack([act[i, combo].mean() for i, combo in enumerate(targets)]).mean()

		loss.backward()
		optimizer.step()

	with torch.no_grad():
		final_rgb = decorrelator.to_rgb(fft_img()).clamp(0, 1)
		np_imgs = (final_rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8") # (B, 3, H, W) -> (B, H, W, 3) (channel re-order)
	return np_imgs

def generate_feature_dataset(
    model: torch.nn.Module,
    features: list[torch.Tensor],
    num_imgs: int,
    device: torch.device,
    savepath: Path,
    save_prefix: str = "",
    batch_size: int = 32,
    img_size: int = 224,
    n_iters: int = 120,
    lr: float = 0.05,
    embedding_size: int = 512,
    max_order: int = 3,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    safe_multiplier: float = 2.0):

    # 5% safety margin to replace discards without re-allocating
    min_std: float = 12.0
    total_combos = int(num_imgs * safe_multiplier) + batch_size
    combos = generate_feature_combinations(
        samples=total_combos, embedding_size=embedding_size, max_order=max_order
    )
    np.random.shuffle(combos)

    dataset_name = f"{save_prefix}_feature_vis.npy" if save_prefix else "feature_vis.npy"
    dataset_savepath = savepath / dataset_name
    dataset_savepath.parent.mkdir(parents=True, exist_ok=True)

    dataset_mmap = np.lib.format.open_memmap(
    	dataset_savepath,
    	dtype="uint8",
    	mode="w+",
    	shape=(num_imgs, img_size, img_size, 3)
    )

    saved = 0
    cursor = 0
    while saved < num_imgs and cursor < len(combos):
        batch = combos[cursor : cursor + batch_size]
        cursor += batch_size

        batch_imgs = generate_feature_samples(
            model=model,
            features=features,
            targets=batch,
            device=device,
            img_size=img_size,
            n_iters=n_iters,
            lr=lr,
            mean=mean,
            std=std,
        )

        # Drop flat/unoptimized images
        valid_mask = batch_imgs.reshape(len(batch), -1).std(axis=1) >= min_std
        valid_imgs = batch_imgs[valid_mask]

        # Write only up to num_imgs
        n_write = min(len(valid_imgs), num_imgs - saved)
        if n_write > 0:
            dataset_mmap[saved : saved + n_write] = valid_imgs[:n_write]
            saved += n_write
            dataset_mmap.flush()

        print(
            f"Saved {saved}/{num_imgs} images (dropped {len(batch) - len(valid_imgs)} in batch)..."
        )

    del dataset_mmap
