import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

def build_wavelet_bank(device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Builds the 40 fixed 3x3 oriented/wavelet filters from Section 3."""
    filters = []

    # Oriented step edges
    base_step = torch.ones((11, 11))
    base_step[:, 5] = 0.0
    base_step[:, 6:] = -1.0

    # High-frequency line filters
    base_line = torch.ones((11, 11))
    base_line[:, 5] = -1.0

    for base in [base_step, base_line]:
        for i in range(8):
            angle = i * 180.0 / 8.0
            # Bilinear rotation of 11x11 kernel via PyTorch grid_sample
            rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rot_mat = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32).unsqueeze(0)
            grid = F.affine_grid(rot_mat, [1, 1, 11, 11], align_corners=False)
            rot = F.grid_sample(base.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=False)[0, 0]
            # Center 3x3 crop zero-mean
            crop = rot[4:7, 4:7]
            crop = crop - crop.mean()
            filters.append(crop)

    # Sparse kernels
    z1 = torch.zeros((3, 3))
    z1[1, 1] = 1.0
    z2 = torch.full((3, 3), -1.0)
    z2[1, 1] = 1.0
    z3 = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
    z4 = torch.sign(z3 - 0.5)
    filters.extend([z1, z2, z3, z4])

    # Inverted wavelets
    bank = torch.stack(filters, dim=0)
    bank = torch.cat([bank, -bank], dim=0)
    return bank.to(device)


def generate_pink_noise(
    height: int,
    width: int,
    alpha: float | None = None,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """Generates a 2D 1/f^alpha noise map with a Laplacian envelope."""
    if alpha is None:
        alpha = float(torch.empty(1).uniform_(0.5, 1.5).item())

    # Frequency grid
    fy = torch.fft.fftfreq(height, device=device).unsqueeze(1)
    fx = torch.fft.rfftfreq(width, device=device).unsqueeze(0)
    radius = torch.sqrt(fx**2 + fy**2)
    radius[0, 0] = 1e-7

    # 1/f^alpha spectral amplitude
    amp = 1.0 / (radius ** alpha)
    amp[0, 0] = 0.0

    # Random Hermitian phase
    phase = torch.empty((height, width // 2 + 1), device=device).uniform_(0, 2 * np.pi)
    spec = amp * torch.exp(1j * phase)
    noise = torch.fft.irfft2(spec, s=(height, width))

    # Laplacian spatial envelope (4x4 upsampled to layer resolution)
    # PyTorch implementation of Laplace distribution: loc=0, scale=1
    u = torch.empty((1, 1, 4, 4), device=device).uniform_(-0.5, 0.5)
    laplace_4x4 = -torch.sign(u) * torch.log(1 - 2 * torch.abs(u) + 1e-7)
    envelope = F.interpolate(laplace_4x4, size=(height, width), mode='bicubic', align_corners=False).squeeze()

    out = noise * envelope
    out = (out - out.mean()) / (out.std() + 1e-7)
    return out

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, up: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.up = up
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1.0 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x: torch.Tensor, styles: torch.Tensor, custom_weight: torch.Tensor | None = None, demodulate: bool = True):
        b, c, h, w = x.shape
        w_base = custom_weight if custom_weight is not None else self.weight
        w_base = w_base * self.scale

        # Modulate: [B, C_out, C_in, K, K]
        weights = w_base.unsqueeze(0) * styles.view(b, 1, c, 1, 1)

        # Demodulate
        if demodulate:
            dcoefs = torch.rsqrt(weights.square().sum(dim=[2, 3, 4]) + 1e-8)  # [B, C_out]
            weights = weights * dcoefs.view(b, -1, 1, 1, 1)

        x = x.view(1, b * c, h, w)
        weights = weights.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)

        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        return x.view(b, self.out_channels, x.shape[-2], x.shape[-1])


class SynthesisLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, w_dim: int = 512, resolution: int = 4, up: bool = False):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.affine = nn.Linear(w_dim, in_channels)
        self.affine.bias.data.fill_(1.0)
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=3, up=up)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise_mode: str = 'random',
        fixed_noise_map: torch.Tensor | None = None,
        wavelet_bank: torch.Tensor | None = None,
        bias_range: float = 0.0,
        randomize_filters: bool = False
    ):
        b = x.shape[0]
        styles = self.affine(w)

        custom_weight = None
        if randomize_filters and wavelet_bank is not None:
            n_sub = np.random.randint(5, len(wavelet_bank) + 1)
            sub_bank = wavelet_bank[torch.randperm(len(wavelet_bank))[:n_sub]]
            chin_inds = torch.randint(n_sub, (self.conv.in_channels,), device=x.device)

            w_wavelet = sub_bank[chin_inds].unsqueeze(0).repeat(self.conv.out_channels, 1, 1, 1)
            scale = torch.randn(self.conv.out_channels, 1, 1, 1, device=x.device)
            custom_weight = w_wavelet * scale

        out = self.conv(x, styles, custom_weight=custom_weight, demodulate=True) # Modulated Convolution

        # Noise Injection part
        if noise_mode == 'random':
            out = out + torch.randn(b, 1, self.resolution, self.resolution, device=x.device) * 0.1
        elif noise_mode == 'pink_random':
            strength = (torch.rand(b, 1, 1, 1, device=x.device) * 0.3)
            if fixed_noise_map is not None:
                # StyleGAN-Oriented: downsample shared 512x512 noise map
                n_map = F.interpolate(fixed_noise_map.view(1, 1, 512, 512), size=(self.resolution, self.resolution), mode='area')
                out = out + n_map * strength
            else:
                p_noise = torch.stack([generate_pink_noise(self.resolution, self.resolution, device=x.device) for _ in range(b)])
                out = out + p_noise.unsqueeze(1) * strength

        b_vec = self.bias
        if bias_range > 0:
            b_vec = torch.empty_like(self.bias).uniform_(-bias_range, bias_range)

        out = out + b_vec.view(1, -1, 1, 1)
        return F.relu(out) * np.sqrt(2.0)


class ToRGBLayer(nn.Module):
    def __init__(self, in_channels: int, img_channels: int = 3, w_dim: int = 512):
        super().__init__()
        self.affine = nn.Linear(w_dim, in_channels)
        self.affine.bias.data.fill_(1.0)
        self.conv = ModulatedConv2d(in_channels, img_channels, kernel_size=1, up=False)
        self.bias = nn.Parameter(torch.zeros(img_channels))

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        styles = self.affine(w)
        # Randomize ToRGB projection at each iteration to increase chromatic diversity
        weight = torch.randn_like(self.conv.weight)
        out = self.conv(x, styles, custom_weight=weight, demodulate=False)
        return out + self.bias.view(1, -1, 1, 1)


class SynthesisBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, resolution: int, w_dim: int = 512, is_first: bool = False):
        super().__init__()
        self.resolution = resolution
        self.is_first = is_first

        if is_first:
            self.const = nn.Parameter(torch.randn(1, out_channels, 4, 4))
            self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim, resolution=4, up=False)
        else:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim, resolution=resolution, up=True)
            self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim, resolution=resolution, up=False)

        self.torgb = ToRGBLayer(out_channels, img_channels=3, w_dim=w_dim)

    def forward(self, x, img, w_iter, noise_mode: str = "random", fixed_noise_map: torch.Tensor | None = None,
        wavelet_bank: torch.Tensor | None = None, bias_range: float = 0.0, randomize_filters: bool = False):
        if self.is_first:
            w_conv = next(w_iter)
            b = w_conv.shape[0]
            x = self.const.repeat(b, 1, 1, 1)
            x = self.conv1(x, w_conv, noise_mode=noise_mode, fixed_noise_map=fixed_noise_map, wavelet_bank=wavelet_bank, bias_range=bias_range, randomize_filters=randomize_filters)
        else:
            x = self.conv0(x, next(w_iter), noise_mode=noise_mode, fixed_noise_map=fixed_noise_map, wavelet_bank=wavelet_bank, bias_range=bias_range, randomize_filters=randomize_filters)
            x = self.conv1(x, next(w_iter), noise_mode=noise_mode, fixed_noise_map=fixed_noise_map, wavelet_bank=wavelet_bank, bias_range=bias_range, randomize_filters=randomize_filters)

        y = self.torgb(x, next(w_iter))
        if img is not None:
            img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
            img = img + y
        else:
            img = y
        return x, img

def batch_normalize_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Vectorized GPU normalization matching natural image channel distributions."""
    b = img.shape[0]
    device = img.device

    # Priors from Learning to See
    mu_prior = torch.tensor([0.483, 0.455, 0.401], device=device)
    mu_scale = torch.tensor([0.145, 0.142, 0.161], device=device)
    std_prior = torch.tensor([0.219, 0.213, 0.213], device=device)
    std_scale = torch.tensor([0.063, 0.062, 0.069], device=device)

    tgt_mu = (torch.randn(b, 3, device=device) * mu_scale + mu_prior).view(b, 3, 1, 1)
    tgt_std = (torch.randn(b, 3, device=device) * std_scale + std_prior).view(b, 3, 1, 1)

    c_mean = img.mean(dim=[-2, -1], keepdim=True)
    c_std = img.std(dim=[-2, -1], keepdim=True) + 1e-8
    norm = ((img - c_mean) / c_std) * tgt_std + tgt_mu

    return (norm.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


class UntrainedStyleGAN(nn.Module):
    def __init__(self, resolution: int = 256, w_dim: int = 512):
        super().__init__()
        self.resolution = resolution
        self.w_dim = w_dim
        self.log2_res = int(np.log2(resolution))
        self.block_resolutions = [2 ** i for i in range(2, self.log2_res + 1)]

        # StyleGAN2 channel schedule: max 512, channel_base = 32768
        channels = {res: min(32768 // res, 512) for res in self.block_resolutions}

        self.blocks = nn.ModuleList()
        self.num_ws = 0
        for res in self.block_resolutions:
            is_first = (res == 4)
            in_c = 0 if is_first else channels[res // 2]
            out_c = channels[res]
            block = SynthesisBlock(in_c, out_c, resolution=res, w_dim=w_dim, is_first=is_first)
            self.num_ws += (2 if is_first else 3) # convs + to_rgb
            self.blocks.append(block)

        self.register_buffer('wavelet_bank', build_wavelet_bank())

    def forward(self, w: torch.Tensor, variant: str = 'oriented') -> torch.Tensor:
        # Variant configuration parameters
        cfg = {
            'random':   {'noise': 'random',      'same_noise': False, 'bias': 0.0, 'filters': False},
            'highfreq': {'noise': 'pink_random', 'same_noise': False, 'bias': 0.0, 'filters': False},
            'sparse':   {'noise': 'pink_random', 'same_noise': False, 'bias': 0.2, 'filters': False},
            'oriented': {'noise': 'pink_random', 'same_noise': True,  'bias': 0.2, 'filters': True},
        }[variant]

        # StyleGAN-Oriented: single pink noise map shared across resolutions
        fixed_noise = generate_pink_noise(512, 512, device=w.device) if cfg['same_noise'] else None

        w_iter = iter(w.unbind(dim=1))
        x, img = None, None
        for block in self.blocks:
            x, img = block(
                x, img, w_iter,
                noise_mode=cfg['noise'],
                fixed_noise_map=fixed_noise,
                wavelet_bank=self.wavelet_bank,
                bias_range=cfg['bias'],
                randomize_filters=cfg['filters']
            )
        return img # type: ignore


def generate_stylegan_dataset(
    savepath: Path,
    device: torch.device,
    n_images: int = 10000,
    resolution: int = 256,
    batch_size: int = 16,
    variant: str = 'oriented',
    reinit_interval: int = 400,
    save_prefix: str = ""
):
    """Generates synthetic representations directly streamed to an open_memmap disk array."""
    dataset_name = f"{save_prefix}_stylegan.npy" if save_prefix else "stylegan.npy"
    dataset_savepath = savepath / dataset_name
    dataset_savepath.parent.mkdir(parents=True, exist_ok=True)

    mmap = np.lib.format.open_memmap(dataset_savepath, mode='w+', dtype=np.uint8, shape=(n_images, resolution, resolution, 3))

    G = UntrainedStyleGAN(resolution=resolution).to(device).eval()
    leak = nn.LeakyReLU(0.2)

    with torch.no_grad():
        written = 0
        while written < n_images:
            current_batch_size = min(batch_size, n_images - written)

            # Periodic network re-initialization matching paper setup
            if (written // batch_size) % (reinit_interval // batch_size) == 0:
                G = UntrainedStyleGAN(resolution=resolution).to(device).eval()

            # Latent parameterization: w ~ LeakyReLU(N(0, I))
            w = leak(torch.randn(current_batch_size, G.num_ws, G.w_dim, device=device))
            raw_img = G(w, variant=variant)

            # Normalize and stream to disk
            mmap[written : written + current_batch_size] = batch_normalize_to_uint8(raw_img)
            written = written + current_batch_size
            mmap.flush()

def generate_stylegan_samples(
    device: torch.device,
    n_samples: int = 8,
    resolution: int = 256,
    variant: str = 'oriented',
) -> np.ndarray:
    """Generates and saves n_samples images directly to output_dir for visual inspection."""

    G = UntrainedStyleGAN(resolution=resolution).to(device).eval()
    leak = nn.LeakyReLU(0.2)

    with torch.no_grad():
        w = leak(torch.randn(n_samples, G.num_ws, G.w_dim, device=device))
        raw = G(w, variant=variant)
        images = batch_normalize_to_uint8(raw)
    return images
