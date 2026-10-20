"""Original code is from https://github.com/mbaradad/learning_with_noise/blob/main/generate_datasets/dead_leaves/generate_dataset.py
Credits to the author there! The paper Learning to See by Looking at Noise (Baraded et al.) Can be found at https://arxiv.org/abs/2106.05963.
StyleGan code was from the paper Training Generative Adversarial Networks with Limited Data https://arxiv.org/abs/2006.06676 and the original
implementation can be found here https://github.com/nvlabs/stylegan2-ada-pytorch
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def build_wavelet_bank(device: torch.device) -> torch.Tensor:
    """Builds a fixed bank of 40 oriented and high-frequency 3x3 wavelet filter kernels.

    Constructs 16 oriented step edges (at 8 distinct angles), 16 high-frequency line filters,
    and 8 sparse point/cross kernels, including positive and negative sign inversions.

    Args:
        device (torch.device, optional): Target device for filter bank tensor. Defaults to CPU.

    Returns:
        torch.Tensor: Filter bank tensor of shape (40, 3, 3).
    """
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
    device: torch.device,
    alpha: float | None = None,
) -> torch.Tensor:
    """Generates a 2D 1/f^alpha pink noise map with a multiscale Laplacian envelope.

    Args:
        height (int): Output noise height.
        width (int): Output noise width.
        alpha (float, optional): Power spectral decay exponent. If None, randomly sampled
            uniformly in [0.5, 1.5]. Defaults to None.
        device (torch.device, optional): Device for tensor computation. Defaults to CPU.

    Returns:
        torch.Tensor: Normalized zero-mean unit-variance 2D noise tensor of shape (height, width).
    """
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
    """Style-modulated and demodulated 2D convolution layer from StyleGAN2.

    Scales convolution weights by incoming per-sample style vectors (modulation) and
    normalizes the output standard deviation to preserve unit variance (demodulation).

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        kernel_size (int, optional): Convolution kernel size. Defaults to 3.
        up (bool, optional): Whether to perform 2x bilinear upsampling before convolution.
            Defaults to False.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, up: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.up = up
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1.0 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(
        self,
        x: torch.Tensor,
        styles: torch.Tensor,
        custom_weight: torch.Tensor | None = None,
        demodulate: bool = True,
    ) -> torch.Tensor:
        """Applies modulated convolution to input feature tensor.

        Args:
            x (torch.Tensor): Input feature tensor of shape (B, C_in, H, W).
            styles (torch.Tensor): Per-sample affine style vectors of shape (B, C_in).
            custom_weight (torch.Tensor, optional): Optional override weight tensor
                (e.g., from wavelet bank). Defaults to None.
            demodulate (bool, optional): Whether to apply variance demodulation. Defaults to True.

        Returns:
            torch.Tensor: Convolved output feature tensor of shape (B, C_out, H_out, W_out).
        """
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
    """StyleGAN2 synthesis sub-layer with affine style modulation and noise injection.

    Args:
        in_channels (int): Input channel dimensionality.
        out_channels (int): Output channel dimensionality.
        w_dim (int, optional): Latent style vector dimension. Defaults to 512.
        resolution (int, optional): Spatial resolution of this layer. Defaults to 4.
        up (bool, optional): Whether to apply 2x spatial upsampling. Defaults to False.
    """

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
        randomize_filters: bool = False,
    ) -> torch.Tensor:
        """Executes forward synthesis pass with optional wavelet filter replacement and noise injection.

        Args:
            x (torch.Tensor): Input activation tensor of shape (B, C_in, H, W).
            w (torch.Tensor): Latent style vector of shape (B, w_dim).
            noise_mode (str, optional): Noise mode ('random' or 'pink_random'). Defaults to 'random'.
            fixed_noise_map (torch.Tensor, optional): Precomputed 512x512 pink noise map for shared noise. Defaults to None.
            wavelet_bank (torch.Tensor, optional): Wavelet filter bank tensor. Defaults to None.
            bias_range (float, optional): Range for uniform random bias perturbation. Defaults to 0.0.
            randomize_filters (bool, optional): Whether to replace weights with sampled wavelets. Defaults to False.

        Returns:
            torch.Tensor: Output activations after LeakyReLU/sqrt(2) scaling of shape (B, C_out, H_out, W_out).
        """
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
    """StyleGAN2 ToRGB projection layer converting hidden features to 3 RGB image channels.

    Args:
        in_channels (int): Input feature channel dimension.
        img_channels (int, optional): Output image color channels. Defaults to 3.
        w_dim (int, optional): Latent style vector dimension. Defaults to 512.
    """

    def __init__(self, in_channels: int, img_channels: int = 3, w_dim: int = 512):
        super().__init__()
        self.affine = nn.Linear(w_dim, in_channels)
        self.affine.bias.data.fill_(1.0)
        self.conv = ModulatedConv2d(in_channels, img_channels, kernel_size=1, up=False)
        self.bias = nn.Parameter(torch.zeros(img_channels))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Projects hidden feature map to RGB space with randomized linear projection weights.

        Args:
            x (torch.Tensor): Feature tensor of shape (B, C, H, W).
            w (torch.Tensor): Style vector of shape (B, w_dim).

        Returns:
            torch.Tensor: RGB image residual tensor of shape (B, 3, H, W).
        """
        styles = self.affine(w)
        # Randomize ToRGB projection at each iteration to increase chromatic diversity
        weight = torch.randn_like(self.conv.weight)
        out = self.conv(x, styles, custom_weight=weight, demodulate=False)
        return out + self.bias.view(1, -1, 1, 1)


class SynthesisBlock(nn.Module):
    """Multi-resolution StyleGAN2 synthesis block containing 2 conv layers and a ToRGB residual.

    Args:
        in_channels (int): Input feature channels.
        out_channels (int): Output feature channels.
        resolution (int): Spatial resolution of this block.
        w_dim (int, optional): Style vector dimension. Defaults to 512.
        is_first (bool, optional): Whether this is the initial 4x4 resolution block. Defaults to False.
    """

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

    def forward(
        self,
        x: torch.Tensor | None,
        img: torch.Tensor | None,
        w_iter,
        noise_mode: str = "random",
        fixed_noise_map: torch.Tensor | None = None,
        wavelet_bank: torch.Tensor | None = None,
        bias_range: float = 0.0,
        randomize_filters: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs block synthesis steps and accumulates RGB output.

        Args:
            x (torch.Tensor, optional): Input feature tensor from previous block.
            img (torch.Tensor, optional): Accumulated RGB image from previous block.
            w_iter: Iterator yielding style vector w for each synthesis step.
            noise_mode (str, optional): Noise mode ('random' or 'pink_random'). Defaults to "random".
            fixed_noise_map (torch.Tensor, optional): Shared 512x512 pink noise map. Defaults to None.
            wavelet_bank (torch.Tensor, optional): Bank of wavelet filter kernels. Defaults to None.
            bias_range (float, optional): Range for uniform bias sampling. Defaults to 0.0.
            randomize_filters (bool, optional): Whether to use wavelet filters. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - x (torch.Tensor): Output feature tensor.
                - img (torch.Tensor): Accumulated RGB image upsampled to block resolution.
        """
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
        return x, img # type: ignore

def batch_normalize_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Vectorized GPU normalization matching natural image channel distributions.

    Standardizes image channels and scales with empirical ImageNet natural scene color
    mean and standard deviation priors from 'Learning to See by Looking at Noise'.

    Args:
        img (torch.Tensor): Raw generated image tensor of shape (B, 3, H, W).

    Returns:
        np.ndarray: Normalized image array of shape (B, H, W, 3) with uint8 values in [0, 255].
    """
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
    """Untrained StyleGAN2 architecture used as an architectural inductive bias generator.

    Generates structured synthetic images without training by passing noise vectors through
    a randomly initialized generative network with weight demodulation, multiscale synthesis,
    and optional wavelet filter/pink noise injection.

    Args:
        resolution (int, optional): Output image spatial resolution. Must be a power of 2.
            Defaults to 256.
        w_dim (int, optional): Dimensionality of latent style vectors. Defaults to 512.
    """

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

        self.register_buffer('wavelet_bank', build_wavelet_bank(device=torch.device("cpu")))

    def forward(self, w: torch.Tensor, variant: str = 'oriented') -> torch.Tensor:
        """Synthesizes images from latent style representations.

        Args:
            w (torch.Tensor): Latent style tensor of shape (B, num_ws, w_dim).
            variant (str, optional): StyleGAN noise configuration variant
                ('random', 'highfreq', 'sparse', or 'oriented'). Defaults to 'oriented'.

        Returns:
            torch.Tensor: Raw generated image tensor of shape (B, 3, resolution, resolution).
        """
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
    save_prefix: str = "",
) -> None:
    """Generates synthetic representations directly streamed to an open_memmap disk array.

    Generates synthetic samples by querying randomly re-initialized StyleGAN networks
    and saving directly into an open_memmap numpy array file to minimize RAM overhead.

    Args:
        savepath (Path): Directory where output dataset will be written.
        device (torch.device): Device on which generator model will run.
        n_images (int, optional): Total number of images to synthesize. Defaults to 10000.
        resolution (int, optional): Image resolution (height and width). Defaults to 256.
        batch_size (int, optional): Batch size per synthesis forward pass. Defaults to 16.
        variant (str, optional): StyleGAN configuration variant ('random', 'highfreq',
            'sparse', or 'oriented'). Defaults to 'oriented'.
        reinit_interval (int, optional): Number of generated samples before reinitializing
            network weights with fresh random parameters. Defaults to 400.
        save_prefix (str, optional): Optional prefix for output filename. Defaults to "".

    Returns:
        None
    """
    dataset_name = f"{save_prefix}_stylegan.npy" if save_prefix else "stylegan.npy"
    dataset_savepath = savepath / dataset_name
    dataset_savepath.parent.mkdir(parents=True, exist_ok=True)

    mmap = np.lib.format.open_memmap(dataset_savepath, mode='w+', dtype=np.uint8, shape=(n_images, resolution, resolution, 3))

    G = UntrainedStyleGAN(resolution=resolution).to(device).eval()
    leak = nn.LeakyReLU(0.2)

    with torch.no_grad():
        written = 0
        samples_since_reinit = 0
        while written < n_images:
            current_batch_size = min(batch_size, n_images - written)

            # Periodic network re-initialization matching paper setup
            if samples_since_reinit >= reinit_interval:
                G = UntrainedStyleGAN(resolution=resolution).to(device).eval()
                samples_since_reinit = 0

            # Latent parameterization: w ~ LeakyReLU(N(0, I))
            w = leak(torch.randn(current_batch_size, G.num_ws, G.w_dim, device=device))
            raw_img = G(w, variant=variant)

            # Normalize and stream to disk
            mmap[written : written + current_batch_size] = batch_normalize_to_uint8(raw_img)
            written += current_batch_size
            samples_since_reinit += current_batch_size
            mmap.flush()

def generate_stylegan_samples(
    device: torch.device,
    n_samples: int = 8,
    resolution: int = 256,
    variant: str = 'oriented',
) -> np.ndarray:
    """Generates synthetic image samples directly in-memory for visual inspection.

    Args:
        device (torch.device): Computing device.
        n_samples (int, optional): Number of synthetic image samples to generate. Defaults to 8.
        resolution (int, optional): Spatial resolution of output images. Defaults to 256.
        variant (str, optional): Noise configuration variant ('random', 'highfreq',
            'sparse', or 'oriented'). Defaults to 'oriented'.

    Returns:
        np.ndarray: Array of generated normalized RGB images of shape (n_samples, resolution, resolution, 3)
            with uint8 values in [0, 255].
    """
    G = UntrainedStyleGAN(resolution=resolution).to(device).eval()
    leak = nn.LeakyReLU(0.2)

    with torch.no_grad():
        w = leak(torch.randn(n_samples, G.num_ws, G.w_dim, device=device))
        raw = G(w, variant=variant)
        images = batch_normalize_to_uint8(raw)
    return images
