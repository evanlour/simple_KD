import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

def colorPCA(ref_image: torch.Tensor):
    B, C, H, W = ref_image.shape
    flat = ref_image.view(B, C, -1)
    meanColor = flat.mean(dim=-1, keepdim=True)
    centered = flat - meanColor
    cov = torch.bmm(centered, centered.transpose(1, 2))
    _, colorPC = torch.linalg.eigh(cov)
    decor = torch.bmm(colorPC.transpose(1, 2), centered)
    return decor.view(B, C, H, W), colorPC, meanColor.view(B, C, 1, 1)

def inv_colorPCA(decor_image: torch.Tensor, colorPC: torch.Tensor, meanColor: torch.Tensor):
    B, C, H, W = decor_image.shape
    rec = torch.bmm(colorPC, decor_image.view(B, C, -1))
    return rec.view(B, C, H, W) + meanColor

def sample_colormodel(B: int, resolution: int, device: torch.device):
    coarse_res = torch.randint(3, 8, (1,)).item()
    coarse_colors = torch.rand(B, 3, coarse_res, coarse_res, device=device) # type: ignore
    ref = F.interpolate(coarse_colors, size=(resolution, resolution), mode='bilinear', align_corners=False)
    ref = F.avg_pool2d(ref, kernel_size=5, stride=1, padding=2)
    return colorPCA(ref)

def match_cumulative_cdf(source: torch.Tensor, target_reference: torch.Tensor) -> torch.Tensor:
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    B, C, H, W = source.shape
    N = H * W
    src_flat = source.view(B, C, N)
    tgt_flat = target_reference.view(B, C, -1)
    M = tgt_flat.shape[-1]

    src_sorted, src_indices = torch.sort(src_flat, dim=-1)
    tgt_sorted, _ = torch.sort(tgt_flat, dim=-1)

    coords = torch.linspace(0, M - 1, N, device=source.device).expand(B, C, N)
    idx_low = coords.long().clamp(max=M - 2)
    idx_high = idx_low + 1
    weight = coords - idx_low

    v_low = torch.gather(tgt_sorted, -1, idx_low)
    v_high = torch.gather(tgt_sorted, -1, idx_high)
    interpolated = (1.0 - weight) * v_low + weight * v_high

    matched = torch.empty_like(src_flat)
    matched.scatter_(-1, src_indices, interpolated)
    return matched.view(B, C, H, W)

def sample_1f_spectrum(B: int, resolution: int, slope_range=(0.5, 3.5), device: torch.device = torch.device('cpu')) -> torch.Tensor:
    H = W = resolution
    s0, s1 = slope_range
    s = torch.empty(B, 1, 1, 1, device=device).uniform_(s0, s1)
    d = torch.randn(B, 1, 1, 1, device=device) * (s1 - s0) / 60.0

    fy = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1)
    fx = torch.fft.fftfreq(W, device=device).view(1, 1, 1, W)
    fr = torch.sqrt(torch.abs(fx) ** (s + d) + torch.abs(fy) ** (s - d))

    fr[:, :, 0, 0] = 1.0
    magnitude = 1.0 / fr
    magnitude[:, :, 0, 0] = 0.0
    return magnitude

def impose_spectrum(x: torch.Tensor, target_magnitude: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True) + eps

    fft_x = torch.fft.fft2(x - mean)
    phase = torch.angle(fft_x)
    unit_phase = torch.polar(torch.ones_like(phase), phase)

    out = torch.fft.ifft2(target_magnitude * unit_phase).real
    out_mean = out.mean(dim=(-2, -1), keepdim=True)
    out_std = out.std(dim=(-2, -1), keepdim=True) + eps
    return (out - out_mean) / out_std * std + mean

def impose_wmm(x: torch.Tensor, filters: torch.Tensor, target_subbands: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    F_num = filters.shape[0]

    fft_x = torch.fft.fft2(x).unsqueeze(2)
    filt = filters.view(1, 1, F_num, H, W)
    coeffs = torch.fft.ifft2(fft_x * filt).real

    coeffs_reshaped = coeffs.permute(0, 2, 1, 3, 4).reshape(B * F_num, C, H, W)
    targets_expanded = target_subbands.unsqueeze(2).expand(B, F_num, C, -1).reshape(B * F_num, C, -1)

    matched = match_cumulative_cdf(coeffs_reshaped, targets_expanded)
    matched_coeffs = matched.view(B, F_num, C, H, W).permute(0, 2, 1, 3, 4)

    matched_coeffs[:, :, 0] = coeffs[:, :, 0]
    matched_coeffs[:, :, -1] = coeffs[:, :, -1]

    fft_matched = torch.fft.fft2(matched_coeffs)
    return torch.sum(torch.fft.ifft2(fft_matched * filt).real, dim=2)

def sample_wmm_model(B: int, num_filters: int, resolution: int, device: torch.device):
    N = resolution * resolution
    r = torch.empty(B, num_filters, 1, device=device).uniform_(0.4, 0.8)
    s = torch.empty(B, num_filters, 1, device=device).uniform_(1.0, 16.0)

    # Exact Generalized Gaussian sampling via Gamma variates
    alpha = 1.0 / r
    gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=torch.ones_like(alpha))
    g = gamma_dist.sample((N,)).squeeze(-1).permute(1, 2, 0) # (B, F, N)
    sign = torch.where(torch.rand(B, num_filters, N, device=device) < 0.5, -1.0, 1.0)
    return sign * s * (g ** (1.0 / r))

def build_steerable_filters(resolution: int, num_scales: int = 3, num_orientations: int = 4, device: torch.device = torch.device('cpu')):
    H = W = resolution
    fy = torch.fft.fftfreq(H, device=device).view(H, 1)
    fx = torch.fft.fftfreq(W, device=device).view(1, W)
    r = torch.sqrt(fx ** 2 + fy ** 2)
    theta = torch.atan2(fy, fx)

    filters = []
    angles = [k * torch.pi / num_orientations for k in range(num_orientations)]
    log_r = torch.log2(r.clamp(min=1e-5) / 0.5)

    hp_mask = (log_r >= 0).float() + (log_r > -1.0).float() * (log_r < 0).float() * torch.cos(torch.pi * log_r / 2) ** 2
    filters.append(torch.sqrt(hp_mask.clamp(0, 1)))

    for s in range(num_scales):
        center = -(s + 1)
        rad_band = torch.cos(torch.pi * (log_r - center).clamp(-1.0, 1.0) / 2) ** 2
        for theta_k in angles:
            ang_diff = torch.remainder(theta - theta_k + torch.pi / 2, torch.pi) - torch.pi / 2
            ang_band = (torch.cos(ang_diff) ** (num_orientations - 1)) * (torch.abs(ang_diff) < torch.pi / 2).float()
            filters.append(torch.sqrt(rad_band * ang_band))

    lp_mask = (log_r <= -num_scales).float() + (log_r > -num_scales) * (log_r < -num_scales + 1) * torch.cos(torch.pi * (log_r + num_scales - 1) / 2) ** 2
    filters.append(torch.sqrt(lp_mask.clamp(0, 1)))

    stack = torch.stack(filters, dim=0)
    norm = torch.sqrt(torch.sum(stack ** 2, dim=0, keepdim=True).clamp(min=1e-12))
    return stack / norm

def generate_statistical_sample(
    B: int,
    resolution: int = 128,
    filters: torch.Tensor | None = None,
    n_iter: int = 10,
    impose_spectrum_flag: bool = True,
    impose_wmm_flag: bool = True,
    impose_color_flag: bool = True,
    correlate_channels: bool = True,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:
    if filters is None and impose_wmm_flag:
        filters = build_steerable_filters(resolution, device=device)

    x = torch.randn(B, 3, resolution, resolution, device=device)

    if impose_color_flag or correlate_channels:
        ref_decor, colorPC, meanColor = sample_colormodel(B, resolution, device=device)
    if impose_spectrum_flag:
        target_mag = sample_1f_spectrum(B, resolution, device=device)
    if impose_wmm_flag:
        target_subbands = sample_wmm_model(B, filters.shape[0], resolution, device=device) # type: ignore

    for _ in range(n_iter):
        if impose_spectrum_flag:
            x = impose_spectrum(x, target_mag) # type: ignore
        if impose_wmm_flag:
            x = impose_wmm(x, filters, target_subbands) # type: ignore
        if impose_color_flag:
            x = match_cumulative_cdf(x, ref_decor) # type: ignore

    if correlate_channels:
        x = inv_colorPCA(x, colorPC, meanColor) # type: ignore

    x_min = x.amin(dim=(-3, -2, -1), keepdim=True)
    x_max = x.amax(dim=(-3, -2, -1), keepdim=True)
    return ((x - x_min) / (x_max - x_min + 1e-8) * 255.0).clamp(0, 255).to(torch.uint8)

def generate_statistical_dataset(
    device: torch.device,
    savepath: Path,
    total_samples: int = 512,
    batch_size: int = 32,
    resolution: int = 128,
    n_iter: int = 10,
    impose_spectrum_flag: bool = True,
    impose_wmm_flag: bool = True,
    impose_color_flag: bool = True,
    correlate_channels: bool = True,
    save_prefix: str = ""
):

    dataset_name = f"{save_prefix}_statistical.npy" if save_prefix else "statistical.npy"
    dataset_savepath = savepath / dataset_name
    dataset_savepath.parent.mkdir(parents=True, exist_ok=True)

    filters = build_steerable_filters(resolution, device=device)

    mmap = np.lib.format.open_memmap(
        dataset_savepath,
        mode='w+',
        dtype=np.uint8,
        shape=(total_samples, resolution, resolution, 3)
    )

    written = 0
    while written < total_samples:
        current_batch = min(batch_size, total_samples - written)
        with torch.no_grad():
            batch = generate_statistical_sample(
                B=current_batch,
                resolution=resolution,
                filters=filters,
                n_iter=n_iter,
                device=device,
                impose_color_flag=impose_color_flag,
                impose_spectrum_flag=impose_spectrum_flag,
                impose_wmm_flag=impose_wmm_flag,
                correlate_channels=correlate_channels
            )
        mmap[written:written + current_batch] = batch.permute(0, 2, 3, 1).cpu().numpy()
        written += current_batch
        print(f"Currently {written} written out of a total of {total_samples}")
    mmap.flush()
