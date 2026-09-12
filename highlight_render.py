import math
import random
import time
from collections import defaultdict
from functools import wraps
from typing import Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Note: transformers imports removed as they are not used
from moge.model.v2 import MoGeModel


# ---------------------------
# helper: smoothstep
def _smoothstep(x, e0, e1):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _resolve_highlight_param(
    value: Union[float, Sequence[float]],
    batch_size: int,
    device: torch.device,
    log_uniform: bool = False,
) -> Union[float, torch.Tensor]:
    """
    Resolve a highlight config value to a scalar or per-sample tensor [B].
    If value is a sequence of two numbers [min, max], sample batch_size values:
    - log_uniform=True: log-uniform in [min, max] (for surface roughness).
    - log_uniform=False: uniform in [min, max] (for intensity).
    Otherwise return the scalar as-is (backward compatible).
    """
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if log_uniform:
            u = torch.rand(batch_size, device=device, dtype=torch.float32)
            log_lo = math.log(lo)
            log_hi = math.log(hi)
            return torch.exp(log_lo + u * (log_hi - log_lo))  # [B]
        u = torch.rand(batch_size, device=device, dtype=torch.float32)
        return u * (hi - lo) + lo  # [B]
    return value


def apply_highlight_falloff(
    H: torch.Tensor,
    falloff: Union[float, torch.Tensor],
    concentration: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Steepen the highlight edge and shrink the soft halo to synthesize
    hard-edged, high-gradient specular highlights (the kind the model currently
    fails to remove). Operates on the per-image-normalized highlight H in [0,1].

    `falloff` in [0, 1): everything below this fraction of the peak is cropped to
    0 and the surviving band [falloff, 1] is rescaled back to [0, 1]. This turns
    the smooth Blinn-Phong lobe into a smaller, crisper highlight with a steep
    intensity gradient at its boundary. falloff == 0 leaves H unchanged (soft
    lobe). `falloff` may be a scalar or a per-sample tensor [B].

    `concentration` >= 0 brightens / saturates the pixels that survive the
    falloff crop by applying a gamma = 1/(1+concentration) power curve to the
    rescaled band, pushing the surviving values up toward 1 (so the kept
    highlight reads as a "more concentrated", brighter core). concentration == 0
    is a no-op (linear rescale, the previous behaviour). `concentration` may be a
    scalar or a per-sample tensor [B].

    H: [B,1,H,W] in [0,1]. Returns [B,1,H,W] in [0,1].
    """

    def _apply_concentration(t: torch.Tensor) -> torch.Tensor:
        # t: [B,1,H,W] rescaled band in [0,1]. gamma<1 brightens the survivors.
        if torch.is_tensor(concentration):
            c = concentration.view(-1, 1, 1, 1).to(t.dtype).clamp_min(0.0)  # [B,1,1,1]
            gamma = 1.0 / (1.0 + c)  # [B,1,1,1]
            return t.clamp_min(0.0).pow(gamma)
        if concentration is None or concentration <= 0.0:
            return t
        gamma = 1.0 / (1.0 + float(concentration))
        return t.clamp_min(0.0).pow(gamma)

    if not torch.is_tensor(falloff):
        if falloff is None or falloff <= 0.0:
            return _apply_concentration(H)
        f = float(falloff)
        t = ((H - f) / max(1.0 - f, 1e-6)).clamp(0.0, 1.0)  # [B,1,H,W]
        return _apply_concentration(t)
    f = falloff.view(-1, 1, 1, 1).to(H.dtype)  # [B,1,1,1]
    t = ((H - f) / (1.0 - f).clamp_min(1e-6)).clamp(0.0, 1.0)  # [B,1,H,W]
    return _apply_concentration(t)


@torch.no_grad()
def make_large_highlight_mask(
    B: int,
    H: int,
    W: int,
    device: torch.device,
    prob: float = 1.0,
    n_blobs: int = 1,
    size_frac: Union[float, Sequence[float]] = (0.12, 0.38),
    elongation: float = 0.5,
    core_frac: float = 0.22,
    peak: Union[float, Sequence[float]] = (0.9, 1.5),
    edge_wobble: float = 0.25,
    warp_scale: int = 24,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate large highlight blobs with a realistic *domed* intensity profile.

    This synthesizes the large contiguous specular morphology (e.g. a wet surface
    under strong light) that the smooth scene-wide Blinn-Phong lobe and the small
    GEOMETRIC_HIGHLIGHTS micro-blobs never produce, so the token inpainter is
    trained on large holes whose interior has no nearby visible context (the
    flat-gray-center failure case).

    The blob is NOT a uniform flat-topped plate (which composites to a frame-wide
    blown-out 255 plateau): it has a small bright core (radius `core_frac` of the
    blob) that falls off smoothly to 0 at the boundary. The per-blob `peak` may
    exceed 1.0 so the core genuinely saturates and clips to pure white over a small
    plateau (real strong glare), while values <1 stay soft — sampling a [min,max]
    that straddles 1.0 gives a realistic mix. The caller composites it with its OWN
    alpha (NOT scaled by the Blinn-Phong `intensity`), so brightness stays
    controlled and a texture gradient survives in the halo.

    Randomness is drawn from the *global* torch RNG (CPU draws for the per-blob
    scalars, a device randn for the border-warp field), so under
    `torch.manual_seed` (FREEZE_VAL_HIGHLIGHTS) the blobs are reproducible and the
    eval metric stays comparable across runs.

    Args:
        B, H, W: batch size and spatial dims of the highlight map to match.
        prob: per-sample probability of receiving a large highlight (else zeros).
        n_blobs: number of blobs per selected sample.
        size_frac: blob diameter as a fraction of min(H,W); scalar or [min,max]
                   (sampled uniformly per blob).
        elongation: 0 = circular, ->1 = increasingly elongated on average.
        core_frac: fraction of the blob *radius* held at full brightness before the
                   smooth falloff begins (small -> tight bright core + wide halo).
        peak: per-blob peak alpha; scalar or [min,max] sampled uniformly. >=1 makes
              the flat core clip to pure white (a saturated plateau); <1 stays soft.
        edge_wobble: amplitude of the irregular border perturbation (0..1). Kept
                     low so the blob stays a single coherent region, not fragments.
        warp_scale: spatial scale (px) of the border perturbation.

    Returns:
        [B,1,H,W] float alpha map (>=0; core may exceed 1 to force a clipped white
        plateau on composite, 0 outside the halo).
    """
    out = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    if isinstance(size_frac, (list, tuple)):
        size_lo, size_hi = float(size_frac[0]), float(size_frac[1])
    else:
        size_lo = size_hi = float(size_frac)
    if isinstance(peak, (list, tuple)):
        peak_lo, peak_hi = float(peak[0]), float(peak[1])
    else:
        peak_lo = peak_hi = float(peak)
    core = min(max(float(core_frac), 0.0), 0.95)
    nb = max(1, int(n_blobs))
    min_hw = float(min(H, W))

    # Base pixel grid (built once, reused for every blob).
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )

    # Host-side scalar draws (CPU RNG is seeded by torch.manual_seed during frozen
    # eval) — one sync instead of one per scalar. Columns: keep, cx, cy, size,
    # axis-ratio, theta, exponent, peak.
    keep = torch.rand(B).tolist()
    rs = torch.rand(B, nb, 7).tolist()

    sc = max(6, int(warp_scale))
    for b in range(B):
        if keep[b] > prob:
            continue
        # Independent border-warp field per sample (device randn -> CUDA RNG).
        h0, w0 = max(2, H // sc), max(2, W // sc)
        warp = F.interpolate(
            torch.randn(1, 2, h0, w0, device=device, dtype=dtype),
            size=(H, W), mode="bicubic", align_corners=False,
        )
        warp = warp / (warp.std(dim=(-2, -1), keepdim=True) + 1e-8)  # [1,2,H,W]

        mask = torch.zeros(H, W, device=device, dtype=dtype)
        for j in range(nb):
            u_cx, u_cy, u_sz, u_ar, u_th, u_p, u_pk = rs[b][j]
            D = max(8.0, (size_lo + (size_hi - size_lo) * u_sz) * min_hw)
            cx, cy = u_cx * W, u_cy * H
            ratio = max(0.25, min(1.0, 1.0 - elongation * u_ar))
            a = D * 0.5
            bax = D * 0.5 * ratio
            theta = u_th * math.pi
            p = 2.0 + 1.4 * u_p  # super-ellipse exponent in [2.0, 3.4]
            pk = peak_lo + (peak_hi - peak_lo) * u_pk
            wob = edge_wobble * 0.35 * D
            Xc = xx + wob * warp[0, 0] - cx
            Yc = yy + wob * warp[0, 1] - cy
            ct, st = math.cos(theta), math.sin(theta)
            xr = ct * Xc + st * Yc
            yr = -st * Xc + ct * Yc
            f = (xr.abs() / (a + 1e-6)).pow(p) + (yr.abs() / (bax + 1e-6)).pow(p)
            # Normalized radius s in [0,1] (s=1 at the boundary). Domed profile:
            # flat peak for s<=core, smooth falloff to 0 at s=1.
            s = f.clamp_min(0.0).pow(1.0 / p)
            fall = ((1.0 - s) / max(1.0 - core, 1e-6)).clamp(0.0, 1.0)
            blob = pk * _smoothstep(fall, 0.0, 1.0)
            mask = torch.maximum(mask, blob)
        out[b, 0] = mask
    return out


@torch.no_grad()
def add_geometric_roughness_torch(
    normals: torch.Tensor,  # [B,3,H,W], in [-1,1] or [0,1]
    # --- blob controls ---
    n_blobs: int = 16,  # number of blobs
    avg_blob_size: float = 0.10,  # avg diameter (fraction of min(H,W) or pixels)
    size_unit: str = "fraction",
    size_spread: float = 0.6,  # lognormal spread; >0 => many small blobs
    elongation_bias: float = 0.6,  # 0: circular, 1: very elongated on avg
    falloff_mean: float = 10,  # mean softness (0=hard edge, 0.5=soft halo)
    falloff_jitter: float = 10,  # variation of falloff per blob
    edge_wobble: float = 0.6,  # amplitude of border perturbation (0..1)
    warp_scale: int = 20,  # spatial scale (px) of border perturbation
    min_separation: float = 0.06,  # keep centers apart (fraction of min(H,W))
    # --- micro-geometry controls (unchanged) ---
    wavelength_px: float = 12.0,
    wavelength_jitter: float = 0.5,
    orientation_anisotropy: float = 0.4,
    octaves: int = 2,
    roughness_strength: float = 10,  # average angular deviation (radians)
    # misc
    seed=None,
    return_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    B, C, H, W = normals.shape
    assert C == 3
    dev = normals.device

    g = torch.Generator(device=dev)
    # if seed is None:
    g.manual_seed(random.randint(0, 1000000))
    # else:
    # g.manual_seed(seed)
    # normalize input to [-1,1]
    n = normals.clone()
    if n.min() >= 0:
        n = n * 2 - 1.0
    n = F.normalize(n, dim=1)

    # base grids
    yy, xx = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )

    # fbm-like noise field for warping (edge perturbations)
    def _smooth_noise(scale=28):
        h0, w0 = max(2, H // scale), max(2, W // scale)
        base = torch.randn(
            1, 2, h0, w0, generator=g, device=dev
        )  # 2 channels -> 2D warp
        field = F.interpolate(base, size=(H, W), mode="bicubic", align_corners=False)
        return field  # [1,2,H,W]

    warp_field = _smooth_noise(scale=max(6, warp_scale))  # shared statistics
    warp_field = warp_field / (
        warp_field.std(dim=(-2, -1), keepdim=True) + 1e-8
    )  # normalize

    # ---------- soft, perturbed super-ellipse blobs ----------
    def make_soft_blob_mask() -> torch.Tensor:
        mask = torch.zeros(1, H, W, device=dev)
        d_mean = (
            (avg_blob_size * min(H, W))
            if size_unit == "fraction"
            else float(avg_blob_size)
        )
        d_mean = max(4.0, d_mean)
        min_sep_px = max(4.0, min_separation * min(H, W))

        centers = []
        tries = 0
        while len(centers) < n_blobs and tries < 6000:
            tries += 1
            cx = torch.empty((), device=dev).uniform_(0, W, generator=g).item()
            cy = torch.empty((), device=dev).uniform_(0, H, generator=g).item()
            if all(
                ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 >= min_sep_px
                for px, py in centers
            ):
                centers.append((cx, cy))

        for cx, cy in centers:
            # sample size from lognormal (many small, some large)
            ln_sigma = size_spread * 0.5  # softer control
            s = torch.exp(
                torch.empty((), device=dev).normal_(0, ln_sigma, generator=g)
            )  # ~lognormal
            D = d_mean * s
            # elongation: sample axis ratio biased to small values
            r = torch.clamp(
                1.0
                - torch.empty((), device=dev).uniform_(0, elongation_bias, generator=g),
                0.15,
                1.0,
            )
            a = D * 0.5  # major semi-axis
            b = D * 0.5 * r  # minor semi-axis
            theta = torch.empty((), device=dev).uniform_(0, 3.1416, generator=g)
            p = torch.empty((), device=dev).uniform_(
                1.8, 3.2, generator=g
            )  # super-ellipse exponent

            # coordinate warp for irregularity
            wob_amp = (
                edge_wobble * 0.35 * D
            )  # scale by size so small blobs aren't shredded
            X = xx + wob_amp * warp_field[:, 0] - cx
            Y = yy + wob_amp * warp_field[:, 1] - cy

            ct, st = torch.cos(theta), torch.sin(theta)
            xr = ct * X + st * Y
            yr = -st * X + ct * Y

            # super-ellipse implicit metric f= (|xr/a|^p + |yr/b|^p)
            f = torch.pow(torch.abs(xr) / (a + 1e-8), p) + torch.pow(
                torch.abs(yr) / (b + 1e-8), p
            )

            # soft falloff via smoothstep around f=1 with randomized width
            soft = (
                falloff_mean
                * (
                    1.0
                    + torch.empty((), device=dev).uniform_(
                        -falloff_jitter, falloff_jitter, generator=g
                    )
                )
            ).clamp(0.05, 0.7)
            edge0 = 1.0 - soft  # inside value to start softening
            edge1 = 1.0 + soft  # outside value where it goes to 0
            blob = (1.0 - _smoothstep(f, edge0, edge1)).clamp(
                0, 1
            )  # 1 at core -> 0 outside
            blob = blob.unsqueeze(0)

            mask = (mask + blob).clamp(0, 1)

        # mild blur to merge micro-holes but keep shapes
        mask = (
            F.gaussian_blur(mask, (5, 5), sigma=(1.2, 1.2))
            if hasattr(F, "gaussian_blur")
            else mask
        )
        return mask  # [1,H,W]

    mask = torch.cat([make_soft_blob_mask() for _ in range(B)], dim=0)  # [B,1,H,W]

    # ---------- micro-geometry (same as before) ----------
    # band-limited Gabor-like height
    yyn, xxn = torch.meshgrid(
        torch.linspace(-1, 1, H, device=dev),
        torch.linspace(-1, 1, W, device=dev),
        indexing="ij",
    )
    height = torch.zeros(B, 1, H, W, device=dev)
    for o in range(octaves):
        lam = wavelength_px * (0.5**o)
        base_freq = 1.0 / max(lam, 1.0)
        dom = torch.empty(B, device=dev).uniform_(0, 3.1416, generator=g)
        theta = dom + torch.empty(B, device=dev).uniform_(
            -3.1416 * orientation_anisotropy * 0.25,
            3.1416 * orientation_anisotropy * 0.25,
            generator=g,
        )
        freq = base_freq * torch.clamp(
            1.0
            + torch.empty(B, device=dev).uniform_(
                -wavelength_jitter, wavelength_jitter, generator=g
            ),
            0.25,
            4.0,
        )
        phase = torch.empty(B, device=dev).uniform_(0, 6.2832, generator=g)
        for b in range(B):
            ct, st = torch.cos(theta[b]), torch.sin(theta[b])
            u = ct * xxn + st * yyn
            height[b : b + 1] += torch.sin(2 * 3.1416 * freq[b] * u + phase[b]) * (
                1.0 / (2**o)
            )
    height = height - height.mean(dim=(-2, -1), keepdim=True)
    height = (
        F.gaussian_blur(height, (5, 5), sigma=(1.0, 1.0))
        if hasattr(F, "gaussian_blur")
        else height
    )
    height = height * mask

    # slopes
    kx = (
        torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=dev, dtype=torch.float32
        ).view(1, 1, 3, 3)
        / 8.0
    )
    ky = (
        torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=dev, dtype=torch.float32
        ).view(1, 1, 3, 3)
        / 8.0
    )
    dhdx = F.conv2d(height, kx, padding=1)
    dhdy = F.conv2d(height, ky, padding=1)

    # TBN from n
    ref = torch.tensor([0.0, 0.0, 1.0], device=dev).view(1, 3, 1, 1).expand(B, -1, H, W)
    parallel = torch.abs((n * ref).sum(1, keepdim=True)) > 0.99
    ref = torch.where(
        parallel, torch.tensor([0.0, 1.0, 0.0], device=dev).view(1, 3, 1, 1), ref
    )
    t = F.normalize(torch.cross(ref, n, dim=1), dim=1)
    b = F.normalize(torch.cross(n, t, dim=1), dim=1)

    # unscaled perturbation from slopes
    offset = (-dhdx) * t + (-dhdy) * b

    # scale so average deviation ≈ roughness_strength (radians)
    test = F.normalize(n + offset, dim=1)
    cosang = (n * test).sum(1).clamp(-1, 1)
    mean_angle = torch.acos(cosang).mean().detach().item() + 1e-8
    scale = roughness_strength / mean_angle
    n_pert = F.normalize(n + offset * scale, dim=1)

    # blend by soft mask
    noisy = F.normalize(torch.lerp(n, n_pert, mask), dim=1)
    return (noisy, mask) if return_mask else (noisy, None)


def time_module(module_name):
    """
    Decorator to time method execution when timing is enabled.

    Args:
        module_name: String identifier for the module being timed

    Usage:
        @time_module("depth_estimation")
        def compute_depth(self, image):
            # method implementation
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "_timing_enabled", False):
                return func(self, *args, **kwargs)

            # Ensure GPU operations are complete before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            try:
                result = func(self, *args, **kwargs)
            finally:
                # Ensure GPU operations are complete after timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
                self.timing_results[module_name].append(elapsed)

            return result

        return wrapper

    return decorator


class HighlightRender(nn.Module):
    def __init__(
        self,
        geometry_model_name="Ruicheng/moge-2-vits-normal",  # Changed parameter name
        height=852,
        width=1096,
        enable_timing=False,
        n_rel=1.5,
        F0=0.4,
    ):
        super().__init__()
        # Load MoGe model for joint depth and normal estimation
        self.geometry_model = MoGeModel.from_pretrained(geometry_model_name)
        self.geometry_model.eval()  # Set to eval mode
        self.height = height
        self.width = width
        self.resizer = transforms.Resize((height, width))

        # Material properties
        self.n_rel = n_rel
        self.F0 = F0

        # Timing functionality
        self._timing_enabled = enable_timing
        self.timing_results = defaultdict(list)

    def enable_timing_mode(self, enabled=True):
        """Enable or disable timing mode."""
        self._timing_enabled = enabled

    def reset_timing_stats(self):
        """Clear all timing statistics."""
        self.timing_results.clear()

    def get_timing_stats(self, detailed=False):
        """
        Get timing statistics for all modules.

        Args:
            detailed: If True, return all measurements. If False, return summary stats.

        Returns:
            dict: Timing statistics in milliseconds
        """
        if not self.timing_results:
            return {}

        if detailed:
            return dict(self.timing_results)

        # Return summary statistics
        stats = {}
        for module_name, times in self.timing_results.items():
            times_array = np.array(times)
            stats[module_name] = {
                "mean_ms": float(np.mean(times_array)),
                "std_ms": float(np.std(times_array)),
                "min_ms": float(np.min(times_array)),
                "max_ms": float(np.max(times_array)),
                "total_ms": float(np.sum(times_array)),
                "count": len(times),
            }
        return stats

    def print_timing_stats(self):
        """Print a formatted timing report."""
        stats = self.get_timing_stats()
        if not stats:
            print("No timing data available. Enable timing mode first.")
            return

        print("\n" + "=" * 60)
        print("HighlightRender Timing Report")
        print("=" * 60)
        print(f"{'Module':<25} {'Mean (ms)':<10} {'Std (ms)':<10} {'Count':<8}")
        print("-" * 60)

        total_time = 0
        for module_name, module_stats in sorted(stats.items()):
            mean_time = module_stats["mean_ms"]
            std_time = module_stats["std_ms"]
            count = module_stats["count"]
            total_time += module_stats["total_ms"]

            print(f"{module_name:<25} {mean_time:<10.2f} {std_time:<10.2f} {count:<8}")

        print("-" * 60)
        print(f"{'Total time':<25} {total_time:<10.2f} ms")
        print("=" * 60)

    def make_pixel_grid(self, B, H, W, device):
        """Return homogeneous pixel grid [B,3,H,W] with x,y in pixels, 1's row."""
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        ones = torch.ones_like(xs, dtype=torch.float32)
        pix = torch.stack([xs.float(), ys.float(), ones], dim=0)  # [3,H,W]
        return pix.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,3,H,W]

    def backproject_depth(self, depth, K):
        """
        Backproject to 3D camera coordinates.
        depth: [B,1,H,W] (meters)
        K:     [B,3,3] intrinsics
        Returns P_cam: [B,3,H,W]
        """
        B, _, H, W = depth.shape
        grid = self.make_pixel_grid(B, H, W, depth.device)  # [B,3,H,W]
        # Sanitize intrinsics and use pseudo-inverse for numerical stability
        K = torch.nan_to_num(K, nan=0.0, posinf=1e6, neginf=-1e6)
        Kinv = torch.linalg.pinv(K)  # [B,3,3]
        rays = (Kinv @ grid.flatten(2)).view(B, 3, H, W)  # [B,3,H,W]
        # Camera looks along +Z; 3D point = ray * depth
        P = rays * depth  # [B,3,H,W]
        return P

    def normalize_vector(self, v, eps=1e-8):
        """Normalize vectors along dim=1"""
        return v / (v.norm(dim=1, keepdim=True).clamp_min(eps))

    def _gaussian_kernel1d(self, sigma: float, dtype, device):
        radius = max(int(3.0 * sigma + 0.5), 1)
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        w = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        w = w / (w.sum() + 1e-12)
        return w.view(1, 1, -1)

    def gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Separable Gaussian blur for [B,C,H,W]. If sigma<=0, returns x.
        """
        if sigma is None or sigma <= 0:
            return x
        B, C, H, W = x.shape
        k1d = self._gaussian_kernel1d(float(sigma), x.dtype, x.device)  # [1,1,K]
        # Horizontal
        pad = k1d.shape[-1] // 2
        w_h = k1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), w_h, groups=C)
        # Vertical
        w_v = k1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), w_v, groups=C)
        return x

    @time_module("depth_estimation")
    @time_module("geometry_estimation")
    def compute_geometry(self, image):
        """
        Compute depth and normals from RGB image using MoGe.

        Args:
            image: [B,3,H,W] RGB image (0-1 normalized)

        Returns:
            depth: [B,1,H,W] depth map
            normals: [B,3,H,W] surface normals
        """
        with torch.no_grad():
            mogeout = self.geometry_model.infer(image)  # image: [B,3,H,W]

        # Extract depth [B,H,W] and normals [B,H,W,3]
        depth = mogeout["depth"]  # [B,H,W]
        normals = mogeout["normal"]  # [B,H,W,3]
        intrinsics = mogeout["intrinsics"].clone()
        intrinsics[:, 0, 2] = image.shape[3] / 2  # cx (width/2)
        intrinsics[:, 1, 2] = image.shape[2] / 2  # cy (height/2)
        intrinsics[:, :2, :2] = intrinsics[:, :2, :2] * 500
        # Sanitize potential NaN/Inf from the geometry model
        depth = torch.nan_to_num(depth, nan=0.0, posinf=1e6, neginf=-1e6)
        normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
        intrinsics = torch.nan_to_num(intrinsics, nan=0.0, posinf=1e6, neginf=-1e6)

        # Resize to target dimensions if needed
        if depth.shape[-2:] != (self.height, self.width):
            depth = self.resizer(depth.unsqueeze(1)).squeeze(1)  # [B,H,W]
            # Resize normals - reshape for resize then back
            normals = normals.permute(0, 3, 1, 2)  # [B,3,H,W]
            normals = self.resizer(normals)  # [B,3,H,W]
            normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
            normals = F.normalize(
                normals, p=2, dim=1, eps=1e-6
            )  # Re-normalize after resize
        else:
            normals = normals.permute(0, 3, 1, 2)  # [B,3,H,W]
            normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
            normals = F.normalize(normals, p=2, dim=1, eps=1e-6)

        # Add channel dimension to depth
        depth = depth.unsqueeze(1)  # [B,1,H,W]

        return depth, normals, intrinsics

    def sample_light_source(
        self,
        dist_to_camera,
        left_right_angle=None,
        above_below_angle=None,
        batch_size=1,
        device="cuda",
    ):
        """
        Sample random light source positions in 3D space using spherical coordinates with intuitive angle controls.
        Camera coordinate system: Z points forward, Y points down, X points right.

        Spherical coordinates: (r, θ, φ) where:
        - r = distance from camera
        - θ = azimuth angle (left-right rotation around Y axis)
        - φ = elevation angle (above-below rotation from horizontal plane)

        Args:
            dist_to_camera: tuple (min_dist, max_dist) - range of distances from camera in meters
            left_right_angle: tuple (min_left, max_right) - horizontal angle range in degrees
                             Negative values = left of camera, positive = right of camera
                             If None, defaults to (-180, 180) degrees (full horizontal range)
            above_below_angle: tuple (min_above, max_below) - vertical angle range in degrees
                              Negative values = above camera, positive = below camera
                              If None, defaults to (-90, 90) degrees (full vertical range)
            batch_size: number of light positions to generate
            device: torch device for the output tensor

        Returns:
            positions: Light source positions in camera space [batch_size,3]
        """
        # Set default angle ranges if not provided
        if left_right_angle is None:
            left_right_angle = (-180, 180)  # Full horizontal range
        if above_below_angle is None:
            above_below_angle = (-90, 90)  # Full vertical range

        # Unpack ranges
        min_dist, max_dist = dist_to_camera
        min_left, max_right = left_right_angle
        min_above, max_below = above_below_angle

        # Sample random values within ranges
        # Distance: uniform sampling
        dist = (
            torch.rand(batch_size, device=device) * (max_dist - min_dist) + min_dist
        )  # [B]

        # Azimuth (left-right): uniform sampling in degrees, then convert to radians
        az_deg = (
            torch.rand(batch_size, device=device) * (max_right - min_left)
            + min_left
            - 90
        )  # [B]
        az_rad = az_deg * (np.pi / 180.0)  # [B]

        # Elevation (above-below): uniform sampling in degrees, then convert to radians
        elev_deg = (
            torch.rand(batch_size, device=device) * (max_below - min_above)
            + min_above
            - 90
        )  # [B]
        elev_rad = elev_deg * (np.pi / 180.0)  # [B]

        # Convert spherical to Cartesian coordinates
        # Camera coordinate system: Z forward, Y down, X right
        x = dist * torch.cos(elev_rad) * torch.sin(az_rad)  # [B] left-right
        y = -dist * torch.sin(
            elev_rad
        )  # [B] above-below (negative for upward elevation)
        z = dist * torch.cos(elev_rad) * torch.cos(az_rad)  # [B] behind-front

        # Stack into position tensor
        positions = torch.stack([x, y, z], dim=-1)  # [B,3]

        return positions

    def compute_fresnel_gamma(self, theta, n_rel=1.5, eps=1e-8):
        """
        Intrinsic DoLP of specular for unpolarized incident light at incidence theta.
        gamma = |Rs - Rp| / (Rs + Rp), with Snell.

        Args:
            theta: [B,1,H,W] incidence angle in radians
            n_rel: relative refractive index n2/n1
            eps: small value for numerical stability

        Returns:
            gamma: [B,1,H,W] degree of linear polarization in [0,1]
        """
        n1 = 1.0
        n2 = n_rel
        sin_t = (n1 / n2) * torch.sin(theta).clamp(-1 + 1e-7, 1 - 1e-7)
        theta_t = torch.asin(sin_t)
        cos_t = torch.cos(theta)
        cos_tt = torch.cos(theta_t)

        # Fresnel reflectances for s (perp) and p (parallel)
        Rs_num = n2 * cos_t - n1 * cos_tt
        Rs_den = n2 * cos_t + n1 * cos_tt
        Rp_num = n1 * cos_t - n2 * cos_tt
        Rp_den = n1 * cos_t + n2 * cos_tt

        Rs = (Rs_num / (Rs_den.clamp_min(eps))) ** 2
        Rp = (Rp_num / (Rp_den.clamp_min(eps))) ** 2

        gamma = (Rs - Rp).abs() / (Rs + Rp + eps)
        return gamma.clamp(0.0, 1.0)

    def schlick_fresnel(self, cos_theta, F0=0.04):
        """
        Scalar Schlick Fresnel for intensity modulation (approximate).
        cos_theta: [B,1,H,W]
        F0: fresnel reflectance at normal incidence
        """
        return F0 + (1 - F0) * (1 - cos_theta).clamp(0, 1) ** 5

    def aop_from_geometry(self, v, l):
        """
        Compute angle of linear polarization (AoLP) for specular:
        polarization dir is perpendicular to the plane of incidence ~ k = v x l

        Args:
            v: [B,3,H,W] surface->camera unit vectors
            l: [B,3,H,W] surface->light unit vectors

        Returns:
            phi: [B,1,H,W] AoLP in radians, reference x-axis is camera +X
        """
        k = torch.cross(v, l, dim=1)  # [B,3,H,W]
        k = self.normalize_vector(k)  # ensure unit
        ex = k[:, 0:1]  # x-component [B,1,H,W]
        ey = k[:, 1:2]  # y-component [B,1,H,W]
        phi = torch.atan2(ey, ex)  # [-pi,pi] [B,1,H,W]
        return phi

    @time_module("viewing_lighting_geometry")
    def compute_viewing_lighting_geometry(self, depth, normals, K, light_pos):
        """
        Compute viewing and lighting geometry for specular reflection.
        Physics: v = P/|P| (view direction), l = (L-P)/|L-P| (light direction), where P are 3D surface points.
        The geometry determines the reflection conditions via the surface normal n.

        Args:
            depth: [B,1,H,W] depth map in meters
            normals: [B,3,H,W] surface normals
            light_pos: [B,3] light positions
            K: [B,3,3] camera intrinsics

        Returns:
            tuple: (v, l, n, nl, nv, light_pos) - view dirs, light dirs, normals, dot products, light positions
        """
        B, _, H, W = depth.shape

        # Reconstruct 3D points and view direction
        P = self.backproject_depth(depth, K)  # [B,3,H,W]
        v = self.normalize_vector(-P)  # [B,3,H,W] surface->camera direction

        # Sample random light position and compute light direction
        L = light_pos.view(B, 3, 1, 1)  # [B,3,1,1]
        l = self.normalize_vector(L - P)  # [B,3,H,W] surface->light direction

        # Normalize surface normals
        n = self.normalize_vector(normals)  # [B,3,H,W]

        # Compute dot products for lighting calculations
        nl = (n * l).sum(1, keepdim=True).clamp_min(0.0)  # cos(theta_l) [B,1,H,W]
        nv = (n * v).sum(1, keepdim=True).clamp_min(0.0)  # cos(theta_v) [B,1,H,W]

        return v, l, n, nl, nv, light_pos, P

    @time_module("blinn_phong_specular")
    def compute_blinn_phong_specular(
        self,
        v,
        l,
        n,
        nv,
        surface_roughness,
        intensity,
        F0,
    ):
        """
        Compute Blinn-Phong specular lobe with Schlick Fresnel approximation.
        Physics: I_spec = intensity * F(θ) * (n·h)^α, where h = (v+l)/|v+l| is the half-vector.
        Fresnel term F(θ) ≈ F0 + (1-F0)(1-cosθ)^5 modulates reflection strength.

        Args:
            v: [B,3,H,W] view directions
            l: [B,3,H,W] light directions
            n: [B,3,H,W] surface normals
            nv: [B,1,H,W] n·v dot product
            surface_roughness: specular exponent α
            intensity: specular strength
            F0: Fresnel reflectance at normal incidence

        Returns:
            H: [B,1,H,W] highlight luminance
        """
        # Compute half-vector
        h = self.normalize_vector(l + v)  # [B,3,H,W] half-vector

        # Compute n·h
        nh = (n * h).sum(1, keepdim=True).clamp(0.0, 1.0)

        # Allow surface_roughness to be scalar or broadcastable tensor
        if torch.is_tensor(surface_roughness):
            if surface_roughness.ndim == 1:
                # [B] -> [B,1,1,1]
                surface_roughness = surface_roughness.view(
                    surface_roughness.shape[0], 1, 1, 1
                )
            elif surface_roughness.ndim == 2:
                # [B,1] -> [B,1,1,1]
                surface_roughness = surface_roughness.view(
                    surface_roughness.shape[0], 1, 1, 1
                )
            # else: assume broadcastable to [B,1,H,W]
        spec_lobe = nh**surface_roughness  # [B,1,H,W]

        # Allow intensity to be scalar or per-sample tensor [B] -> [B,1,1,1]
        intensity_bc = intensity
        if torch.is_tensor(intensity):
            if intensity.ndim == 1:
                intensity_bc = intensity.view(intensity.shape[0], 1, 1, 1)
            elif intensity.ndim == 2:
                intensity_bc = intensity.view(intensity.shape[0], 1, 1, 1)

        # Apply Schlick Fresnel approximation
        F = self.schlick_fresnel(nv, F0=F0)  # [B,1,H,W]
        H = intensity_bc * F * spec_lobe  # [B,1,H,W]
        return H

    @time_module("polarization_parameters")
    def compute_polarization_parameters(self, v, l, nl, n_rel):
        """
        Compute polarization parameters from Fresnel reflection theory.
        Physics: DoLP γ = |Rs-Rp|/(Rs+Rp) from Fresnel equations, AoLP φ from plane of incidence.
        The plane of incidence is spanned by incident and reflected rays, perpendicular to v×l.

        Args:
            v: [B,3,H,W] view directions
            l: [B,3,H,W] light directions
            nl: [B,1,H,W] n·l dot product
            n_rel: relative refractive index

        Returns:
            tuple: (gamma_spec, phi_spec) - degree and angle of linear polarization
        """
        # Compute incidence angle and intrinsic DoLP from Fresnel theory
        theta = torch.acos(nl.clamp(-1 + 1e-7, 1 - 1e-7))  # [B,1,H,W]
        gamma_spec = self.compute_fresnel_gamma(theta, n_rel=n_rel)  # [B,1,H,W]

        # Compute AoLP from geometry (plane of incidence)
        phi_spec = self.aop_from_geometry(v, l)  # [B,1,H,W]

        return gamma_spec, phi_spec

    @time_module("highlight_stokes_vector")
    def compute_highlight_stokes_vector(self, H, gamma_spec, phi_spec):
        """
        Construct Stokes vector for specular highlight.
        Physics: S = [S0, S1, S2] where S1 = γS0cos(2φ), S2 = γS0sin(2φ).
        This represents partially linearly polarized light with DoLP γ and AoLP φ.

        Args:
            H: [B,1,H,W] highlight intensity (S0)
            gamma_spec: [B,1,H,W] degree of linear polarization
            phi_spec: [B,1,H,W] angle of linear polarization

        Returns:
            tuple: (S0_H, S1_H, S2_H) - highlight Stokes components
        """
        S0_H = H  # total highlight intensity [B,1,H,W]
        c2p = torch.cos(2.0 * phi_spec)  # [B,1,H,W]
        s2p = torch.sin(2.0 * phi_spec)  # [B,1,H,W]
        S1_H = gamma_spec * S0_H * c2p  # [B,1,H,W]
        S2_H = gamma_spec * S0_H * s2p  # [B,1,H,W]

        return S0_H, S1_H, S2_H

    def update_stokes_with_highlight(self, stokes, S0_H, S1_H, S2_H):
        """
        Add highlight contribution to existing Stokes parameters.
        Physics: Incoherent addition S_total = S_scene + S_highlight for independent sources.
        Each Stokes component adds linearly for incoherent superposition.

        Args:
            stokes: [B,3,H,W] input Stokes parameters (S0,S1,S2)
            S0_H, S1_H, S2_H: [B,1,H,W] highlight Stokes components

        Returns:
            S_new: [B,3,H,W] updated Stokes parameters
        """
        S0, S1, S2 = stokes[:, 0:1], stokes[:, 1:2], stokes[:, 2:3]  # [B,1,H,W] each
        S0_new = S0 + S0_H  # [B,1,H,W]
        S1_new = S1 + S1_H  # [B,1,H,W]
        S2_new = S2 + S2_H  # [B,1,H,W]
        S_new = torch.cat([S0_new, S1_new, S2_new], dim=1)  # [B,3,H,W]

        return S_new

    def update_rgb_with_highlight(self, rgb, H, intensity=1.0):
        """
        Compose highlight over existing RGB image using H as both intensity and alpha.

        Args:
            rgb: [B,3,H,W] input RGB image, all values in [0,1]
            H: [B,1,H,W] single-channel highlight map; both highlight intensity and blending alpha, in [0,1]
            intensity: scalar or [B]/[B,1,1,1] tensor; scales highlight strength per sample.

        Returns:
            [B,3,H,W]: RGB result with highlights composed, values clamped to [0,1].
        """
        # Per-sample intensity: ensure broadcastable [B,1,1,1]
        if torch.is_tensor(intensity):
            if intensity.ndim == 1:
                intensity = intensity.view(intensity.shape[0], 1, 1, 1)
            elif intensity.ndim == 2:
                intensity = intensity.view(intensity.shape[0], 1, 1, 1)
        # Use H as an alpha map to blend pure white highlight over input rgb, modulated by intensity
        alpha = (H * intensity).clamp(0, 1)  # [B,1,H,W]
        alpha_rgb = alpha.expand_as(rgb)  # [B,3,H,W]
        # Traditional "over" operator: composed = (1-a)*rgb + a*white
        composed = rgb * (1.0 - alpha_rgb) + alpha_rgb * 1.0
        composed = composed.clamp(0, 1)
        return composed

    @time_module("highlight_synthesis")
    def synthesize_highlight_with_stokes(
        self,
        rgb_lin,
        stokes,
        depth,
        normals,
        K,
        light_pos=None,
        surface_roughness=64.0,
        intensity=1.0,
        clamp_H=True,
        falloff=0.0,
        concentration=0.0,
    ):
        """
        Synthesize specular highlights and build the corresponding Stokes vector.

        Args:
            rgb_lin: [B,3,H,W] linear RGB (0-1)
            stokes: [B,3,H,W] input Stokes (S0,S1,S2); only used for shape/context
            depth: [B,1,H,W] metric depth (meters)
            normals: [B,3,H,W] unit surface normals
            K: [B,3,3] camera intrinsics
            light_pos: Optional [B,3] light positions (camera coordinates); if None, sampled
            surface_roughness: Base Blinn-Phong exponent α (higher = sharper highlight)
            intensity: Scalar specular scale
            clamp_H: If True, normalizes each image's highlight to [0,1]

        Returns:
            H: [B,1,H,W] highlight luminance
            H_stokes: [B,3,H,W] highlight Stokes (S0,S1,S2)
            H_aop: [B,1,H,W] angle of polarization (radians)
            H_dop: [B,1,H,W] degree of linear polarization [0,1]
            light_pos: [B,3] light positions used
            pcloud: [B,3,H,W] reconstructed 3D points
            l: [B,3,H,W] light directions
            v: [B,3,H,W] view directions
        """
        B, _, H, W = rgb_lin.shape
        device = rgb_lin.device
        # 0) Sample random light positions (one per image)
        if light_pos is None:
            light_pos = self.sample_light_source(
                dist_to_camera=(0.3, 1),  # 0.3m to 1m from camera
                left_right_angle=(-110, 110),  # 110° left to 110° right
                above_below_angle=(-90, 90),  # 90° above to 90° below
                batch_size=B,
                device=device,
            )  # [B,3]
        # 1) Compute viewing and lighting geometry
        v, l, n, nl, nv, light_pos, pcloud = self.compute_viewing_lighting_geometry(
            depth, normals, K, light_pos #* torch.tensor([-1, -1, 1]).to(device)
        )

        # 2) Compute Blinn-Phong specular lobe with Fresnel modulation
        H = self.compute_blinn_phong_specular(
            v,
            l,
            n,
            nv,
            surface_roughness,
            intensity,
            self.F0,
        )
        if clamp_H:
            H = H / (
                H.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            )  # normalize to [0,1]

        # 3b) Optionally steepen the falloff to synthesize hard-edged,
        # high-gradient highlights (operates on the normalized lobe in [0,1]).
        # `concentration` additionally brightens the surviving core.
        H = apply_highlight_falloff(H, falloff, concentration)

        # 4) Compute polarization parameters from Fresnel theory
        H_dop, H_aop = self.compute_polarization_parameters(v, l, nl, self.n_rel)

        # 5) Build highlight Stokes vector
        S0_H, S1_H, S2_H = self.compute_highlight_stokes_vector(H, H_dop, H_aop)
        H_stokes = torch.cat([S0_H, S1_H, S2_H], dim=1)

        return H, H_stokes, H_aop, H_dop, light_pos, pcloud, l, v

    # @time_module("forward_pass")
    def forward(
        self,
        rgb,
        pol=None,
        light_pos=None,
        intrinsic="compute",
        surface_roughness=80.0,
        intensity=10.0,
        highlight_falloff=0.0,
        highlight_density=0.0,
        # Large contiguous highlight parameters (direct injection into H; see
        # make_large_highlight_mask). large_highlight_prob=0 disables (default).
        large_highlight_prob=0.0,
        large_highlight_count=1,
        large_highlight_size=(0.12, 0.38),
        large_highlight_elongation=0.5,
        large_highlight_core_frac=0.22,
        large_highlight_peak=(0.7, 1.0),
        large_highlight_edge_wobble=0.25,
        # Geometric roughness parameters
        n_blobs=0,
        avg_blob_size=0.10,
        size_unit="fraction",
        size_spread=0.6,
        elongation_bias=0.6,
        falloff_mean=100,
        falloff_jitter=10,
        edge_wobble=1,
        warp_scale=18,
        min_separation=0.06,
        wavelength_px=1.0,
        wavelength_jitter=0.5,
        orientation_anisotropy=0.4,
        octaves=2,
        roughness_strength=1.5,
        seed=0,
        return_dataset_highlights=True,
        dataset_highlight_dilation=None,
        dataset_highlight_threshold=None,
        dataset_highlight_use_luminance=False,
    ):
        """
        Forward pass for polar highlight synthesis with geometric roughness.

        Args:
            rgb: [B,3,H,W] RGB image (0-1 normalized)
            pol: Optional [B,3,H,W] Stokes (S0,S1,S2); if provided, returns stokes_highlighted
            light_pos: Optional [B,3] light positions; sampled if None
            intrinsic: [B,3,3] intrinsics or "compute" to use MoGe intrinsics
            surface_roughness: Base Blinn-Phong exponent α (float or [min, max]; if range, sampled log-uniform per sample)
            intensity: Specular strength multiplier (float or [min, max]; if range, sampled uniform per sample)
            highlight_falloff: Edge-hardness crop in [0,1) (float or [min,max]; uniform per sample). 0 = soft lobe.
            highlight_density: Brightness boost (>=0) of the pixels that survive the falloff crop
                                     (float or [min,max]; uniform per sample). Applies a gamma=1/(1+c) curve
                                     to the rescaled band, pushing survivors toward 1. 0 = no boost.
            # Geometric roughness parameters
            n_blobs: number of blobs
            avg_blob_size: avg diameter (fraction of min(H,W) or pixels)
            size_unit: "fraction" or "pixels"
            size_spread: lognormal spread; >0 => many small blobs
            elongation_bias: 0: circular, 1: very elongated on avg
            falloff_mean: mean softness (0=hard edge, 0.5=soft halo)
            falloff_jitter: variation of falloff per blob
            edge_wobble: amplitude of border perturbation (0..1)
            warp_scale: spatial scale (px) of border perturbation
            min_separation: keep centers apart (fraction of min(H,W))
            wavelength_px: wavelength in pixels
            wavelength_jitter: variation in wavelength
            orientation_anisotropy: anisotropy factor
            octaves: number of octaves for micro-geometry
            roughness_strength: average angular deviation (radians)
            seed: random seed
            return_dataset_highlights: Whether to compute dataset highlight masks
            dataset_highlight_dilation: Dilation kernel size for dataset highlights (optional)
            dataset_highlight_threshold: Brightness/luminance threshold for dataset highlights (optional)
            dataset_highlight_use_luminance: If True, use perceptually-weighted luminance for dataset highlights;
                                            if False, use simple mean brightness (default: False)

        Returns:
            dict with keys:
                'highlight': [B,1,H,W]
                'rgb_highlighted': [B,3,H,W]
                'stokes_highlight': [B,3,H,W]
                'stokes_highlighted': Optional [B,3,H,W] if pol provided
                'depth': [B,1,H,W]
                'normals': [B,3,H,W]
                'H_dop': [B,1,H,W]
                'H_aop': [B,1,H,W]
                'intrinsic': [B,3,3]
                'light_pos': [B,3]
                'pcloud': [B,3,H,W]
                'light_dir': [B,3,H,W]
                'view_dir': [B,3,H,W]
                'surface_roughness': float or [B] tensor, resolved value(s) used for specular exponent
                'intensity': float or [B] tensor, resolved value(s) used for specular strength
        """
        # Ensure tensors are on GPU
        device = rgb.device
        rgb = rgb.to(device)
        if pol is not None:
            pol = pol.to(device)

        B, C, H, W = rgb.shape

        # Resolve highlight params: scalar or [min, max] -> per-sample tensor or scalar (sampling done here)
        surface_roughness = _resolve_highlight_param(
            surface_roughness, B, device, log_uniform=True
        )
        intensity = _resolve_highlight_param(
            intensity, B, device, log_uniform=False
        )
        # Falloff / edge-hardness: scalar or [min,max] (uniform per-sample). 0 = soft lobe.
        falloff = _resolve_highlight_param(
            highlight_falloff, B, device, log_uniform=False
        )
        # Concentration / brightness boost of the surviving core: scalar or
        # [min,max] (uniform per-sample). 0 = linear rescale (no boost).
        concentration = _resolve_highlight_param(
            highlight_density, B, device, log_uniform=False
        )

        # If intensity is scalar zero, skip all highlight and geometry computations (tensor => never skip)
        if not torch.is_tensor(intensity) and intensity == 0:

            def zeros_1hw(ch):
                return torch.zeros((B, ch, H, W), device=device, dtype=rgb.dtype)

            def zeros_b3():
                return torch.zeros((B, 3), device=device, dtype=rgb.dtype)

            def zeros_b33():
                return torch.zeros((B, 3, 3), device=device, dtype=rgb.dtype)

            result = {
                "highlight": zeros_1hw(1),
                "rgb_highlighted": rgb,
                "stokes_highlight": zeros_1hw(3),
                "depth": zeros_1hw(1),
                "normals": zeros_1hw(3),
                "H_dop": zeros_1hw(1),
                "H_aop": zeros_1hw(1),
                "intrinsic": zeros_b33(),
                "light_pos": zeros_b3(),
                "pcloud": zeros_1hw(3),
                "light_dir": zeros_1hw(3),
                "view_dir": zeros_1hw(3),
                "stokes_highlighted": zeros_1hw(3) if pol is not None else None,
                "supervision_mask": torch.ones_like(zeros_1hw(1)),
                "pixel_supervision_mask": torch.ones_like(zeros_1hw(1)),
                "surface_roughness": surface_roughness,
                "intensity": intensity,
                "highlight_falloff": falloff,
                "highlight_density": concentration,
            }
            if (
                return_dataset_highlights
                and dataset_highlight_threshold is not None
                and dataset_highlight_dilation is not None
            ):
                result["dataset_highlights_soft_mask"] = get_soft_highlight_map(
                    rgb,
                    threshold=dataset_highlight_threshold,
                    use_luminance=dataset_highlight_use_luminance,
                )
            return result

        depth, normals, moge_intrinsics = self.compute_geometry(
            rgb
        )  # [B,1,H,W], [B,3,H,W]

        # Apply geometric roughness to normals
        if n_blobs > 0:
            normals, _ = add_geometric_roughness_torch(
                normals,
                n_blobs=n_blobs,
                avg_blob_size=avg_blob_size,
                size_unit=size_unit,
                size_spread=size_spread,
                elongation_bias=elongation_bias,
                falloff_mean=falloff_mean,
                falloff_jitter=falloff_jitter,
                edge_wobble=edge_wobble,
                warp_scale=warp_scale,
                min_separation=min_separation,
                wavelength_px=wavelength_px,
                wavelength_jitter=wavelength_jitter,
                orientation_anisotropy=orientation_anisotropy,
                octaves=octaves,
                roughness_strength=roughness_strength,
                seed=seed,
                return_mask=False,
            )

        if intrinsic == "compute":
            intrinsic = moge_intrinsics.to(device)
        else:
            intrinsic = intrinsic.to(device)

        # 3) Synthesize highlights and update Stokes parameters
        # Use dummy pol if not provided for the synthesis function
        pol_input = pol if pol is not None else torch.zeros_like(rgb)
        
        # Light position gets rescaled in relation to the average depth
        light_pos_rescaled = light_pos * depth.mean()
        
        H, H_stokes, H_aop, H_dop, light_pos_random, pcloud, light_dir, view_dir = (
            self.synthesize_highlight_with_stokes(
                rgb,
                pol_input,
                depth,
                normals,
                intrinsic,
                light_pos=light_pos_rescaled,
                surface_roughness=surface_roughness,
                intensity=intensity,
                falloff=falloff,
                concentration=concentration,
            )
        )

        # 3c) Large contiguous highlights (the morphology the smooth Blinn-Phong
        # lobe never produces). `big` is a domed alpha map; we composite it with its
        # OWN alpha AFTER the lobe composite (NOT scaled by `intensity`), so its
        # brightness stays controlled instead of being driven to a blown-out 255
        # plateau. The highlight map records max(H, big) so the downstream inpaint /
        # supervision masks cover the whole blob and the token inpainter is trained
        # to reconstruct large-hole interiors against the clean tokens.
        big = None
        if large_highlight_prob and float(large_highlight_prob) > 0.0:
            big = make_large_highlight_mask(
                B,
                H.shape[-2],
                H.shape[-1],
                device,
                prob=float(large_highlight_prob),
                n_blobs=int(large_highlight_count),
                size_frac=large_highlight_size,
                elongation=float(large_highlight_elongation),
                core_frac=float(large_highlight_core_frac),
                peak=large_highlight_peak,
                edge_wobble=float(large_highlight_edge_wobble),
                dtype=H.dtype,
            )

        rgb_highlighted = self.update_rgb_with_highlight(rgb, H, intensity=intensity)
        if big is not None:
            # "over" composite of a white highlight using the blob's own alpha.
            rgb_highlighted = (rgb_highlighted * (1.0 - big) + big).clamp(0, 1)
            H = torch.clamp(torch.maximum(H, big), 0, 1)

        # 4) Update scene Stokes parameters with highlight contribution (only if pol provided)
        result = {
            "highlight": H,
            "rgb_highlighted": rgb_highlighted,
            "stokes_highlight": H_stokes,
            "depth": depth,
            "normals": normals,
            "H_dop": H_dop,
            "H_aop": H_aop,
            "intrinsic": intrinsic,
            "light_pos": light_pos_random if light_pos is None else light_pos,
            "pcloud": pcloud,
            "light_dir": light_dir,
            "view_dir": view_dir,
            "surface_roughness": surface_roughness,
            "intensity": intensity,
            "highlight_falloff": falloff,
            "highlight_density": concentration,
        }

        # Dataset Highlights are thresholded here
        if (
            return_dataset_highlights
            and dataset_highlight_threshold is not None
            and dataset_highlight_dilation is not None
        ):
            dataset_highlights_soft_mask = get_soft_highlight_map(
                rgb,
                threshold=dataset_highlight_threshold,
                use_luminance=dataset_highlight_use_luminance,
            )
            # dataset_highlights_bool_mask = (dataset_highlights_soft_mask > 0).int()
            result["dataset_highlights_soft_mask"] = dataset_highlights_soft_mask
            # Fix: Check soft mask > 0 (or a small epsilon) to respect the original threshold logic.
            # The previous logic (soft > threshold) effectively applied the threshold twice.
            highlight_region = (
                dataset_highlights_soft_mask > 0.01
            )

            # 3. Dilate highlights (expand the forbidden area)
            if (
                dataset_highlight_dilation is not None
                and dataset_highlight_dilation > 0
            ):
                highlight_region = (
                    torch.nn.functional.max_pool2d(
                        highlight_region.float(),
                        kernel_size=dataset_highlight_dilation,
                        stride=1,
                        padding=dataset_highlight_dilation // 2,
                    )
                    > 0
                )

            # 4. SUPERVISION mask = everywhere except highlights
            result["pixel_supervision_mask"] = (~highlight_region).int()
        # Only compute stokes_highlighted if pol was provided
        if pol is not None:
            S0_H, S1_H, S2_H = H_stokes[:, 0:1], H_stokes[:, 1:2], H_stokes[:, 2:3]
            stokes_updated = self.update_stokes_with_highlight(pol, S0_H, S1_H, S2_H)
            result["stokes_highlighted"] = stokes_updated
        else:
            result["stokes_highlighted"] = None

        return result


def get_soft_highlight_map(
    rgb_image: torch.Tensor, threshold: float = 0.7, use_luminance: bool = False
) -> torch.Tensor:
    """
    Create a soft map of highlights from an RGB image.

    Returns 0 for brightness/luminance <= threshold, 1 for brightness/luminance == 1,
    with smooth linear interpolation in between.

    Args:
        rgb_image: Input RGB image tensor of shape [B, 3, H, W] with values in [0, 1]
        threshold: Threshold value for highlight detection (default: 0.7)
        use_luminance: If True, use perceptually-weighted luminance (0.299*R + 0.587*G + 0.114*B);
                      if False, use simple mean brightness (default: False)

    Returns:
        Soft highlight map tensor of shape [B, 1, H, W] with values in [0, 1]
    """
    # Compute brightness/luminance
    if use_luminance:
        # Perceptually-weighted luminance: 0.299*R + 0.587*G + 0.114*B
        luminance_weights = torch.tensor(
            [0.299, 0.587, 0.114], dtype=rgb_image.dtype, device=rgb_image.device
        ).view(1, 3, 1, 1)
        brightness = (rgb_image * luminance_weights).sum(
            dim=1, keepdim=True
        )  # Shape: [B, 1, H, W]
    else:
        # Simple mean brightness across RGB channels
        brightness = rgb_image.mean(dim=1, keepdim=True)  # Shape: [B, 1, H, W]

    # Linear interpolation from (threshold, 0) to (1, 1)
    # For brightness > threshold: (brightness - threshold) / (1 - threshold)
    # For brightness <= threshold: 0
    soft_highlights = (brightness - threshold) / (1 - threshold)
    soft_highlights = torch.clamp(
        soft_highlights, min=0.0
    )  # Clamp negative values to 0

    return soft_highlights  # Shape: [B, 1, H, W]
