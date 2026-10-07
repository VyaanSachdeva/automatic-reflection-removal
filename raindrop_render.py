"""
Virtual Raindrop-Glare Synthesis (VRGS) -- torch-batched renderer.

Batched, GPU-capable rewrite of vrgs_prototype.py, modeled after
UnReflectAnything's highlight_render.py (HighlightRender): operates on
[B,3,H,W] tensors with per-sample randomized parameters, so it can slot into
engine.py's training loop the same way HighlightRender does -- called once
per training step on a batch of clean RGB images pulled from the dataloader.

Physical model: see vrgs_prototype.py's module docstring (Roser & Geiger
geometric-optics droplet model, simplified to a radial wide-FOV-compression
remap rather than full Snell's-law ray tracing; glare reuses Flare7K assets).
This file replaces vrgs_prototype.py's per-image PIL/numpy loops with
per-droplet full-image torch.nn.functional.grid_sample warps, blended via
soft masks -- the same vectorization style highlight_render.py uses.

Realism refinements beyond the first prototype: explicit defocus blur on
droplets (windshield droplets sit only cm from the lens, well outside the
scene's focal plane -- grid_sample's bilinear resampling alone isn't enough
blur), a lognormal droplet size distribution (many small, few large -- not
uniform), one shared wind/motion direction per image driving all streaks
(not independently-random per droplet), core darkening (water absorbs/
scatters light, so the interior should read darker than the bright rim),
brightness-weighted glare placement (glare occurs where a real light source
sits in the scene, not at a uniformly random position), and a global
low-alpha "wet glass" haze pass on top of the discrete droplets.

NOT yet wired into engine.py: this only defines the renderer + a standalone
tensor-batch smoke test. Wiring into the training loop needs reconciling
HighlightRender's highlight-specific kwargs (surface_roughness, light_pos,
DATASET_HIGHLIGHT_* thresholds) with this module's rain-specific ones
(rain_intensity, capture_context) -- deliberately kept separate for now
rather than forcing a fake shared interface.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from raindrop_paths import RAINDROP_ROOT

FLARE_ROOT = RAINDROP_ROOT / "extracted" / "Flare7K" / "Flare7Kpp" / "Flare7K" / "Scattering_Flare"
FLARE_CATEGORIES = ["Streak", "Glare_with_shimmer", "Light_Source", "Core"]
MAX_DROPLETS = 20


class RaindropRender(nn.Module):
    """Synthesizes water-on-glass degradation (droplet occlusion + trailing
    streak + light glare) on a batch of clean RGB images in [0,1], [B,3,H,W].
    """

    def __init__(self, height: int, width: int, flare_root: Path = FLARE_ROOT, max_flare_cache: int = 64):
        super().__init__()
        self.height = height
        self.width = width
        # Weighted toward "Streak" (a clean light-trail look) over the other
        # categories, which lean toward colorful ring/halo lens-flare shapes
        # -- not what a wet-glass light streak actually looks like. Also
        # desaturated + warm-tinted at composite time (see _get_flare_tensor).
        streak_paths = list((flare_root / "Streak").glob("*.png"))
        other_paths = [p for cat in ["Glare_with_shimmer", "Light_Source", "Core"] for p in (flare_root / cat).glob("*.png")]
        if not streak_paths and not other_paths:
            raise FileNotFoundError(f"No flare assets found under {flare_root}")
        random.shuffle(streak_paths)
        random.shuffle(other_paths)
        n_streak = min(len(streak_paths), int(max_flare_cache * 0.7))
        n_other = max_flare_cache - n_streak
        self._flare_paths = streak_paths[:n_streak] + other_paths[:n_other]
        self._flare_cache: dict[int, torch.Tensor] = {}

    def _get_flare_tensor(self, idx: int, device: torch.device) -> torch.Tensor:
        if idx not in self._flare_cache:
            img = Image.open(self._flare_paths[idx]).convert("RGB")
            t = TF.to_tensor(img)  # [3,h,w] in [0,1]
            # Desaturate (keep luminance only) then tint warm-white, so this
            # reads as a plain bright light streak rather than a multicolor
            # chromatic-aberration flare artifact.
            luminance = (t[0] * 0.299 + t[1] * 0.587 + t[2] * 0.114).unsqueeze(0)
            warm_tint = torch.tensor([1.0, 0.95, 0.85]).view(3, 1, 1)
            t = luminance * warm_tint
            self._flare_cache[idx] = t.clamp(0, 1)  # CPU cache
        return self._flare_cache[idx].to(device)

    @staticmethod
    def _droplet_grid(
        cx: torch.Tensor, cy: torch.Tensor, radius: torch.Tensor, H: int, W: int, fov_factor: float = 3.0
    ) -> torch.Tensor:
        """Build a full-image sampling grid (for F.grid_sample) implementing
        the radial wide-FOV-compression remap within `radius` px of (cx,cy)
        (pixel coords), identity elsewhere. cx,cy,radius: scalars (python
        float/int or 0-d tensors) for one droplet on one image."""
        device = cx.device if isinstance(cx, torch.Tensor) else torch.device("cpu")
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        dx = xx - cx
        dy = yy - cy
        dist = torch.sqrt(dx**2 + dy**2)
        d = torch.clamp(dist / radius, 0, 1)
        theta = torch.atan2(dy, dx)
        inside = dist <= radius
        src_r = radius * (d.clamp_min(1e-6) ** 0.4) * fov_factor
        src_x = torch.where(inside, cx + src_r * torch.cos(theta), xx)
        src_y = torch.where(inside, cy + src_r * torch.sin(theta), yy)
        # normalize to [-1,1] for grid_sample
        grid_x = (src_x / (W - 1)) * 2 - 1
        grid_y = (src_y / (H - 1)) * 2 - 1
        return torch.stack([grid_x, grid_y], dim=-1)  # [H,W,2]

    @staticmethod
    def _feathered_circle_mask(cx: torch.Tensor, cy: torch.Tensor, radius: torch.Tensor, H: int, W: int, feather: float = 0.15) -> torch.Tensor:
        device = cx.device if isinstance(cx, torch.Tensor) else torch.device("cpu")
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        d = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
        return torch.clamp((1.0 - d) / feather, 0, 1)  # [H,W]

    def _rim_highlight(self, cx, cy, radius, H, W, device) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        nx = (xx - cx) / radius
        ny = (yy - cy) / radius
        r2 = nx**2 + ny**2
        inside = r2 <= 1
        nz = torch.sqrt(torch.clamp(1 - r2, min=0))
        light_dir = torch.tensor([-0.4, -0.6, 0.7], device=device)
        light_dir = light_dir / light_dir.norm()
        ndotl = torch.clamp(nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2], 0, 1)
        fresnel_like = (1 - nz) ** 2
        highlight = torch.clamp(ndotl**6 * (230 / 255) + fresnel_like * (90 / 255), 0, 1)
        return torch.where(inside, highlight, torch.zeros_like(highlight))  # [H,W], add to all 3 channels

    @staticmethod
    def _box_blur(img: torch.Tensor, factor: int) -> torch.Tensor:
        """Cheap defocus blur via downsample+upsample. img: [C,H,W]. Windshield
        droplets sit only a few cm from the lens, far closer than the focal
        plane, so real ones are noticeably out of focus -- grid_sample's
        bilinear resampling alone isn't nearly enough blur on its own."""
        if factor <= 1:
            return img
        c, h, w = img.shape
        small = F.avg_pool2d(img.unsqueeze(0), factor, factor, ceil_mode=True)
        return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)[0]

    @staticmethod
    def _box_blur_batch(img: torch.Tensor, factor: int) -> torch.Tensor:
        """Batched version of _box_blur for [B,3,H,W]."""
        if factor <= 1:
            return img
        b, c, h, w = img.shape
        small = F.avg_pool2d(img, factor, factor, ceil_mode=True)
        return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)

    @staticmethod
    def _sample_lognormal_radius(min_r: int, max_r: int) -> int:
        """Real rain has many small droplets and comparatively few large
        ones -- a lognormal (mode near min_r, long tail toward max_r) reads
        far more natural than uniform sampling across the full range."""
        mu, sigma = math.log(min_r * 1.3), 0.5
        r = int(random.lognormvariate(mu, sigma))
        return max(min_r, min(max_r, r))

    def _apply_one_droplet(self, img: torch.Tensor, cx: int, cy: int, radius: int, device) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """img: [3,H,W], updated in place. Returns (soft_mask [h,w], box (y0, y1, x0, x1))
        where the mask covers img[:, y0:y1, x0:x1].

        Works on a crop around the droplet rather than the full frame: the warp
        reads at most fov_factor * radius px from the centre, and the blend mask
        is zero beyond `radius`, so nothing outside the crop can change. Cost
        scales with the droplet's area instead of the image's, which is what
        made the full-frame version slow (~200 full-frame passes per image)."""
        H, W = img.shape[-2:]
        reach = int(math.ceil(radius * 3.0)) + 2  # fov_factor in _droplet_grid, plus a bilinear margin
        y0, y1 = max(0, cy - reach), min(H, cy + reach + 1)
        x0, x1 = max(0, cx - reach), min(W, cx + reach + 1)
        crop = img[:, y0:y1, x0:x1]
        h, w = crop.shape[-2:]
        cx_t = torch.tensor(float(cx - x0), device=device)
        cy_t = torch.tensor(float(cy - y0), device=device)
        r_t = torch.tensor(float(radius), device=device)
        H, W = h, w  # everything below is in crop coordinates
        grid = self._droplet_grid(cx_t, cy_t, r_t, H, W).unsqueeze(0)  # [1,h,w,2]
        warped = F.grid_sample(crop.unsqueeze(0), grid, align_corners=True, mode="bilinear")[0]  # [3,h,w]
        blur_factor = max(1, int(radius * 0.12))
        warped = self._box_blur(warped, blur_factor)
        # Core darkening: water absorbs/scatters some light, so the droplet
        # interior should read slightly darker and less saturated than the
        # sharp bright rim, not just a blurry copy of the background.
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        d_norm = (torch.sqrt((xx - cx_t) ** 2 + (yy - cy_t) ** 2) / r_t).clamp(0, 1)
        core_darken = 1.0 - 0.2 * (1.0 - d_norm).clamp(0, 1).unsqueeze(0)
        gray = warped.mean(dim=0, keepdim=True)
        warped = (warped * 0.85 + gray * 0.15) * core_darken
        warped = warped + self._rim_highlight(cx_t, cy_t, r_t, H, W, device).unsqueeze(0)
        warped = warped.clamp(0, 1)
        mask = self._feathered_circle_mask(cx_t, cy_t, r_t, H, W)
        img[:, y0:y1, x0:x1] = crop * (1 - mask) + warped * mask
        return mask, (y0, y1, x0, x1)

    def forward(
        self,
        rgb: torch.Tensor,
        rain_intensity: float,
        capture_context: str = "dashcam",
        return_mask: bool = True,
    ) -> dict:
        """rgb: [B,3,H,W] in [0,1]. rain_intensity: 0..1 (scalar, applied to whole batch
        for simplicity -- per-sample randomization happens in droplet count/placement).
        Returns dict with 'degraded' [B,3,H,W] and 'mask' [B,1,H,W] (soft union of
        droplet/streak/glare regions)."""
        B, _, H, W = rgb.shape
        device = rgb.device
        out = rgb.clone()
        mask_accum = torch.zeros(B, H, W, device=device)

        n_drops = int(10 + rain_intensity * 40)
        min_r, max_r = int(min(H, W) * 0.02), int(min(H, W) * 0.09)

        for b in range(B):
            img = out[b]
            # One shared wind/motion direction per image: all droplets on the
            # same windshield in the same moment slide the same way (shared
            # airflow / vehicle motion), not independently-random directions.
            base_angle = 90.0 if capture_context != "dashcam" else random.uniform(75, 105)

            for _ in range(n_drops):
                radius = self._sample_lognormal_radius(max(min_r, 1), max(max_r, min_r + 1))
                cx = random.randint(radius, W - radius - 1)
                cy = random.randint(radius, H - radius - 1)
                drop_mask, (y0, y1, x0, x1) = self._apply_one_droplet(img, cx, cy, radius, device)
                mask_accum[b, y0:y1, x0:x1] = torch.clamp(mask_accum[b, y0:y1, x0:x1] + drop_mask, 0, 1)

                if random.random() < 0.7 * rain_intensity + 0.35:
                    angle = math.radians(base_angle + random.uniform(-12, 12))
                    length = int(radius * random.uniform(2, 6))
                    steps = max(3, length // max(radius, 4))
                    for i in range(steps):
                        t = i / steps
                        r2 = int(radius * (1 - 0.6 * t))
                        if r2 < 3:
                            break
                        sx = int(cx + math.cos(angle) * length * t)
                        sy = int(cy + math.sin(angle) * length * t)
                        if not (r2 <= sx < W - r2 and r2 <= sy < H - r2):
                            break
                        streak_mask, (y0, y1, x0, x1) = self._apply_one_droplet(img, sx, sy, r2, device)
                        mask_accum[b, y0:y1, x0:x1] = torch.clamp(mask_accum[b, y0:y1, x0:x1] + streak_mask * (1 - 0.7 * t), 0, 1)

            # Bias glare placement toward pixels already bright in the clean
            # frame -- glare/bloom happens where a real light source sits in
            # the scene (headlight, streetlight, taillight), not anywhere at
            # random. Falls back toward uniform when nothing stands out.
            luminance = (rgb[b, 0] * 0.299 + rgb[b, 1] * 0.587 + rgb[b, 2] * 0.114)
            weights = (luminance.clamp_min(0.05) ** 4).flatten()
            weights = weights / weights.sum()

            n_flares = int(1 + rain_intensity * 4)
            for _ in range(n_flares):
                flare = self._get_flare_tensor(random.randrange(len(self._flare_paths)), device)
                scale = random.uniform(0.5, 1.8) * (W / 1440)
                fh, fw = int(flare.shape[1] * scale), int(flare.shape[2] * scale)
                if fh < 4 or fw < 4:
                    continue
                flare_r = F.interpolate(flare.unsqueeze(0), size=(fh, fw), mode="bilinear", align_corners=False)[0]
                intensity = 1.0 + 0.8 * rain_intensity
                flat_idx = torch.multinomial(weights, 1).item()
                cy, cx = divmod(flat_idx, W)
                x0, y0 = cx - fw // 2, cy - fh // 2
                fx0, fy0 = max(0, -x0), max(0, -y0)
                ox0, oy0 = max(0, x0), max(0, y0)
                ox1, oy1 = min(W, x0 + fw), min(H, y0 + fh)
                if ox1 <= ox0 or oy1 <= oy0:
                    continue
                region = flare_r[:, fy0 : fy0 + (oy1 - oy0), fx0 : fx0 + (ox1 - ox0)] * intensity
                img[:, oy0:oy1, ox0:ox1] = torch.clamp(img[:, oy0:oy1, ox0:ox1] + region, 0, 1)
                flare_mask = (region.mean(dim=0) > 0.02).float()
                mask_accum[b, oy0:oy1, ox0:ox1] = torch.clamp(mask_accum[b, oy0:oy1, ox0:ox1] + flare_mask, 0, 1)

            out[b] = img

        # Global wet-glass veiling haze: beyond the discrete droplets, a
        # sheet of water/mist across the whole windshield lowers contrast
        # and saturation slightly overall -- proportional to rain_intensity,
        # applied everywhere (not masked to droplet regions).
        if rain_intensity > 0:
            hazy = self._box_blur_batch(out, factor=max(1, int(min(H, W) * 0.03)))
            gray = out.mean(dim=1, keepdim=True)
            hazy = hazy * 0.7 + gray * 0.3
            haze_alpha = 0.12 * rain_intensity
            out = torch.clamp(out * (1 - haze_alpha) + hazy * haze_alpha + haze_alpha * 0.05, 0, 1)

        result = {"degraded": out}
        if return_mask:
            result["mask"] = mask_accum.unsqueeze(1)  # [B,1,H,W]
        return result


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kitti_dir = RAINDROP_ROOT / "extracted" / "kitti_2011_09_26_drive_0001"
    out_dir = RAINDROP_ROOT / "vrgs_preview_batched"
    out_dir.mkdir(exist_ok=True)

    images = sorted(kitti_dir.rglob("*.png"))[:4]
    tensors = []
    for p in images:
        img = Image.open(p).convert("RGB").resize((704, 224))  # KITTI is wide; keep aspect-ish, small for a fast smoke test
        tensors.append(TF.to_tensor(img))
    batch = torch.stack(tensors).to(device)  # [4,3,224,704]

    renderer = RaindropRender(height=batch.shape[-2], width=batch.shape[-1]).to(device)
    print(f"Loaded {len(renderer._flare_paths)} flare assets, running on {device}")

    for intensity in [0.3, 0.6, 1.0]:
        result = renderer(batch, rain_intensity=intensity, capture_context="dashcam")
        degraded, mask = result["degraded"], result["mask"]
        print(f"intensity={intensity}: degraded {degraded.shape} range [{degraded.min():.3f},{degraded.max():.3f}], mask {mask.shape} mean={mask.mean():.3f}")
        for b in range(batch.shape[0]):
            clean_img = TF.to_pil_image(batch[b].cpu())
            deg_img = TF.to_pil_image(degraded[b].cpu())
            mask_img = TF.to_pil_image(mask[b].cpu())
            side = Image.new("RGB", (clean_img.width * 3, clean_img.height))
            side.paste(clean_img, (0, 0))
            side.paste(deg_img, (clean_img.width, 0))
            side.paste(mask_img.convert("RGB"), (clean_img.width * 2, 0))
            side.save(out_dir / f"batch{b}_intensity{intensity:.1f}.png")
    print(f"Saved previews to {out_dir}")
