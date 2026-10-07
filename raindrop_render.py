"""
Virtual Raindrop-Glare Synthesis (VRGS) -- torch-batched renderer.

Batched, GPU-capable rewrite of vrgs_prototype.py, modeled after
UnReflectAnything's highlight_render.py (HighlightRender): operates on
[B,3,H,W] tensors with per-sample randomized parameters, so it slots into
engine.py's training loop the same way HighlightRender does (see
raindrop_engine.py) -- called once per training step on a batch of clean RGB
images pulled from the dataloader.

Physical model: Roser & Geiger's geometric-optics droplet model, simplified to
a radial remap rather than full Snell's-law ray tracing: a drop acts as a tiny
wide-angle lens, so it shows a shrunken, upside-down view of the scene around
it. Glare reuses Flare7K assets.

What real water on a windshield or camera dome looks like, and how each drop
here is drawn to match: the glass sits a few cm from a lens focused far away,
so drops are out of focus (Gaussian defocus, soft edges) and mostly
see-through (a per-drop opacity, with only a faint rim and core darkening);
they are irregular (rotated ellipses with a wobbly edge), numerous and mostly
small (a lognormal size distribution with a long tail toward big drops), and
cut off at the frame edge. Each drop draws its own look, so a frame mixes
faint and clearly visible drops. Larger drops slide, leaving streaks that share
one wind/motion direction per image; glare is placed where the scene already
has bright light sources; a low-alpha "wet glass" haze goes over everything.
Each image in a batch gets its own rain intensity.

The returned mask marks droplets, streaks and strong glare (the detector's
training target); very faint drops count for less. Haze is global and not in
the mask.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from raindrop_paths import RAINDROP_ROOT

FLARE_ROOT = RAINDROP_ROOT / "extracted" / "Flare7K" / "Flare7Kpp" / "Flare7K" / "Scattering_Flare"
FLARE_CATEGORIES = ["Streak", "Glare_with_shimmer", "Light_Source", "Core"]
MAX_DROPLETS = 20
FOV_FACTOR = 3.0  # how far around itself a drop "sees", in radii
GLARE_MASK_THRESHOLD = 0.3  # glare brighter than this counts as degraded; fainter halo does not


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
    def _box_blur_batch(img: torch.Tensor, factor: int) -> torch.Tensor:
        """Cheap blur via downsample+upsample, for the global haze. img: [B,3,H,W]."""
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
        mu, sigma = math.log(min_r * 1.5), 0.55
        r = int(random.lognormvariate(mu, sigma))
        return max(min_r, min(max_r, r))

    @staticmethod
    def _sample_look() -> dict:
        """One drop's appearance. Most drops are faint and see-through; a few
        are dense enough to hide what's behind them."""
        return {
            "alpha": 0.25 + 0.75 * random.betavariate(2.5, 1.5),  # opacity, mean ~0.7
            "rim": random.uniform(0.05, 0.45),  # specular rim strength (was fixed and bright)
            "dark": random.uniform(0.0, 0.15),  # core darkening
            "blur": random.uniform(0.04, 0.45),  # defocus sigma, as a fraction of the radius
            "feather": random.uniform(0.15, 0.6),  # width of the soft edge, as a fraction of the radius
            "aspect": random.uniform(0.7, 1.0),  # ellipse minor/major axis
            "angle": random.uniform(0, math.pi),
            "wobble": random.uniform(0.0, 0.12),  # edge irregularity
            "lobes": random.choice((2, 3, 4)),
            "phase": random.uniform(0, 2 * math.pi),
        }

    def _apply_one_droplet(self, img: torch.Tensor, cx: int, cy: int, radius: int, device, look: dict
                           ) -> tuple[torch.Tensor, tuple[int, int, int, int]] | None:
        """img: [3,H,W], updated in place. Returns (mask [h,w], box (y0, y1, x0, x1))
        where the mask covers img[:, y0:y1, x0:x1], or None if the drop is
        entirely off-frame.

        Works on a crop around the droplet rather than the full frame: the warp
        reads at most FOV_FACTOR * radius px from the centre, and the drop's
        shape is zero beyond (1 + wobble) * radius, so nothing outside the crop
        can change. Cost scales with the droplet's area instead of the image's."""
        H, W = img.shape[-2:]
        reach = int(math.ceil(radius * FOV_FACTOR)) + 2
        y0, y1 = max(0, cy - reach), min(H, cy + reach + 1)
        x0, x1 = max(0, cx - reach), min(W, cx + reach + 1)
        if y1 - y0 < 2 or x1 - x0 < 2:
            return None
        crop = img[:, y0:y1, x0:x1]
        h, w = crop.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        dx, dy = xx - (cx - x0), yy - (cy - y0)

        # Shape: a rotated ellipse whose edge wobbles a little. d is the
        # distance from the centre in units of the local edge radius (1 = edge).
        ca, sa = math.cos(look["angle"]), math.sin(look["angle"])
        u, v = dx * ca + dy * sa, -dx * sa + dy * ca
        edge = radius * (1 + look["wobble"] * torch.sin(look["lobes"] * torch.atan2(v, u) + look["phase"]))
        d = torch.sqrt(u**2 + (v / look["aspect"]) ** 2) / edge
        inside = d <= 1

        # Refraction: the drop shows a shrunken, upside-down view of the scene
        # around it (sampled on the opposite side of the centre).
        theta = torch.atan2(dy, dx)
        src_r = radius * FOV_FACTOR * d.clamp(1e-6, 1) ** 0.4
        src_x = torch.where(inside, (cx - x0) - src_r * torch.cos(theta), xx)
        src_y = torch.where(inside, (cy - y0) - src_r * torch.sin(theta), yy)
        grid = torch.stack([src_x / (w - 1) * 2 - 1, src_y / (h - 1) * 2 - 1], dim=-1).unsqueeze(0)
        warped = F.grid_sample(crop.unsqueeze(0), grid, align_corners=True, mode="bilinear",
                               padding_mode="reflection")[0]  # [3,h,w]

        # Defocus: the glass is far closer than the focal plane.
        sigma = max(0.5, radius * look["blur"])
        k = min(2 * math.ceil(2.5 * sigma) + 1, 2 * min(h, w) - 1)
        k = k if k % 2 else k - 1
        if k >= 3:
            warped = TF.gaussian_blur(warped, [k, k], [sigma, sigma])

        # Water absorbs/scatters a little light (slightly darker, less
        # saturated core) and a faint specular rim catches the light.
        dc = d.clamp(0, 1)
        warped = warped * 0.85 + warped.mean(dim=0, keepdim=True) * 0.15
        warped = warped * (1.0 - look["dark"] * (1.0 - dc))
        nz = torch.sqrt((1 - dc**2).clamp_min(0))
        nx, ny = dc * torch.cos(theta), dc * torch.sin(theta)
        ndotl = (nx * -0.4 + ny * -0.6 + nz * 0.7) / math.sqrt(0.4**2 + 0.6**2 + 0.7**2)
        rim = (ndotl.clamp(0, 1) ** 6 * 0.9 + (1 - nz) ** 2 * 0.35) * look["rim"]
        warped = (warped + torch.where(inside, rim, torch.zeros_like(rim))).clamp(0, 1)

        # Soft, defocused edge; blended by the drop's opacity.
        shape = ((1.0 - d) / look["feather"]).clamp(0, 1)
        alpha = shape * look["alpha"]
        img[:, y0:y1, x0:x1] = crop * (1 - alpha) + warped * alpha
        # Mask target: the drop's footprint, counting very faint drops for less.
        return shape * min(1.0, look["alpha"] / 0.3), (y0, y1, x0, x1)

    def forward(
        self,
        rgb: torch.Tensor,
        rain_intensity: float | Sequence[float],
        capture_context: str = "dashcam",
        return_mask: bool = True,
    ) -> dict:
        """rgb: [B,3,H,W] in [0,1]. rain_intensity: 0..1, one value for the whole
        batch or one per image.
        Returns dict with 'degraded' [B,3,H,W] and 'mask' [B,1,H,W] (soft union of
        droplet/streak/glare regions)."""
        B, _, H, W = rgb.shape
        device = rgb.device
        intensities = [float(rain_intensity)] * B if isinstance(rain_intensity, (int, float)) else [float(x) for x in rain_intensity]
        out = rgb.clone()
        mask_accum = torch.zeros(B, H, W, device=device)

        # Sized from the frame's area, not its short side: KITTI frames are
        # wide and short, and drops on the glass don't care about the crop.
        side = math.sqrt(H * W)
        min_r = max(3, int(side * 0.012))
        max_r = max(min_r + 1, int(side * 0.06))

        def add_mask(b, m, box, weight=1.0):
            y0, y1, x0, x1 = box
            mask_accum[b, y0:y1, x0:x1] = torch.clamp(mask_accum[b, y0:y1, x0:x1] + m * weight, 0, 1)

        for b in range(B):
            ri = intensities[b]
            img = out[b]
            # One shared wind/motion direction per image: all droplets on the
            # same windshield in the same moment slide the same way (shared
            # airflow / vehicle motion), not independently-random directions.
            base_angle = 90.0 if capture_context != "dashcam" else random.uniform(75, 105)

            for _ in range(int(25 + ri * 175)):
                radius = self._sample_lognormal_radius(min_r, max_r)
                # Drops may sit partly off the frame edge.
                cx = random.randint(-radius // 2, W - 1 + radius // 2)
                cy = random.randint(-radius // 2, H - 1 + radius // 2)
                look = self._sample_look()
                res = self._apply_one_droplet(img, cx, cy, radius, device, look)
                if res is not None:
                    add_mask(b, *res)

                # Only larger drops are heavy enough to slide and leave a trail.
                if radius >= 2 * min_r and random.random() < 0.15 + 0.35 * ri:
                    angle = math.radians(base_angle + random.uniform(-12, 12))
                    length = int(radius * random.uniform(2, 6))
                    steps = max(3, length // max(radius, 4))
                    trail = {**look, "alpha": look["alpha"] * 0.8}
                    for i in range(steps):
                        t = i / steps
                        r2 = int(radius * (1 - 0.6 * t))
                        if r2 < 2:
                            break
                        sx = int(cx + math.cos(angle) * length * t)
                        sy = int(cy + math.sin(angle) * length * t)
                        if not (0 <= sx < W and 0 <= sy < H):
                            break
                        res = self._apply_one_droplet(img, sx, sy, r2, device, trail)
                        if res is not None:
                            add_mask(b, *res, weight=1 - 0.7 * t)

            # Bias glare placement toward pixels already bright in the clean
            # frame -- glare/bloom happens where a real light source sits in
            # the scene (headlight, streetlight, taillight), not anywhere at
            # random. Falls back toward uniform when nothing stands out.
            luminance = (rgb[b, 0] * 0.299 + rgb[b, 1] * 0.587 + rgb[b, 2] * 0.114)
            weights = (luminance.clamp_min(0.05) ** 4).flatten()
            weights = weights / weights.sum()

            # Smaller and rarer than before: on a daytime frame the brightest
            # spot is usually the sky, and big flares there blew out the frame.
            n_flares = int(random.random() < 0.6) + int(ri * 2)
            for _ in range(n_flares):
                flare = self._get_flare_tensor(random.randrange(len(self._flare_paths)), device)
                scale = random.uniform(0.3, 0.9) * (W / 1440)
                fh, fw = int(flare.shape[1] * scale), int(flare.shape[2] * scale)
                if fh < 4 or fw < 4:
                    continue
                flare_r = F.interpolate(flare.unsqueeze(0), size=(fh, fw), mode="bilinear", align_corners=False)[0]
                intensity = 0.6 + 0.6 * ri
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
                # Only the bright part of the glare counts as degraded: the faint
                # halo around it is barely visible, and marking it taught the
                # detector to flag large areas around bright sky.
                flare_mask = (region.mean(dim=0) > GLARE_MASK_THRESHOLD).float()
                add_mask(b, flare_mask, (oy0, oy1, ox0, ox1))

        # Global wet-glass veiling haze: beyond the discrete droplets, a
        # sheet of water/mist across the whole windshield lowers contrast
        # and saturation slightly overall -- proportional to rain_intensity,
        # applied everywhere (not masked to droplet regions).
        haze_alpha = 0.12 * torch.tensor(intensities, device=device).view(B, 1, 1, 1)
        if haze_alpha.max() > 0:
            hazy = self._box_blur_batch(out, factor=max(1, int(min(H, W) * 0.03)))
            gray = out.mean(dim=1, keepdim=True)
            hazy = hazy * 0.7 + gray * 0.3
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
