"""
Runs the stock pretrained UnReflectAnything model one stage at a time, so the
demo can show what it's doing while it runs: encode -> detect -> choose
regions -> fill in feature space -> decode.

Mirrors the installed package exactly -- its inference preprocessing
(square resize to the encoder's image_size, resize back afterwards) and
UnReflect_Model_TokenInpainter.forward in site-packages/models.py -- so the
final image matches `ura inference`. The package ships its own top-level
`models`/`token_inpainters` modules, which this repo's files of the same name
would shadow (the repo is first on sys.path), so they're loaded with
site-packages temporarily put first.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from raindrop_paths import RAINDROP_ROOT

# The GPU when there is one; RAIN_DEVICE=cpu forces the CPU (benchmark_efficiency.py does).
DEVICE = torch.device(os.environ.get("RAIN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
_wrapper = None
_M = None  # the package's models module, for its mask helpers
_load_lock = threading.Lock()


def _load():
    """Safe to call from a background preload thread and a request at once."""
    global _wrapper, _M
    with _load_lock:
        if _wrapper is not None:
            return
        site_pkgs = next(p for p in sys.path if p.rstrip("\\/").endswith("site-packages"))
        saved_path = sys.path[:]
        for name in ("models", "token_inpainters"):
            sys.modules.pop(name, None)
        sys.path.insert(0, site_pkgs)
        try:
            import unreflectanything as ura

            wrapper = ura.model(pretrained=True, device=str(DEVICE))
            wrapper.eval()
            _M = sys.modules["models"]
            _wrapper = wrapper
        finally:
            sys.path[:] = saved_path


def _heatmap(t: torch.Tensor, cmap: str = "inferno") -> Image.Image:
    # Square-root scaled: the detector's likelihoods are mostly small, and a
    # linear map renders them as near-black on a projector.
    a = np.sqrt(np.clip(t.squeeze().float().cpu().numpy(), 0, 1))
    rgba = matplotlib.colormaps[cmap](a)
    return Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))


def _overlay(rgb: torch.Tensor, mask: torch.Tensor, color=(1.0, 0.15, 0.15), alpha: float = 0.55) -> Image.Image:
    m = mask.squeeze().float().cpu().clamp(0, 1).unsqueeze(0)
    tint = torch.tensor(color).view(3, 1, 1)
    img = rgb.squeeze(0).cpu()
    return TF.to_pil_image((img * (1 - alpha * m) + tint * alpha * m).clamp(0, 1))


STAGES = [
    "Encode the image with DINOv3",
    "Detect regions to remove",
    "Choose patches to repair",
    "Fill in those patches in feature space",
    "Decode back to an image",
]


def ground_truth_mask(rainy: Image.Image, clean: Image.Image, thresh: float = 0.06) -> torch.Tensor:
    """Where a rainy photo differs from its clean ground truth: the actual
    droplets. [1,1,H,W] bool at the photo's own resolution. Only possible for
    paired data (DeRaindrop), so it's a diagnostic -- "what if detection were
    right" -- not something available on arbitrary footage."""
    r = TF.to_tensor(rainy.convert("RGB"))
    g = TF.to_tensor(clean.convert("RGB").resize(rainy.size))
    d = (r - g).abs().mean(0, keepdim=True).unsqueeze(0)
    return F.avg_pool2d(d, 5, 1, 2) > thresh


def neighbour_fill(T: torch.Tensor, hole: torch.Tensor, grid: int, smooth_passes: int = 10) -> torch.Tensor:
    """Fills hole tokens from the visible tokens around them, peeling inward one
    ring at a time (each new token = mean of its already-known 3×3 neighbours),
    then smooths inside the holes so fills aren't blocky. T: [B,N,C],
    hole: [B,N] bool (True = fill). No learned weights.

    Used instead of the released TokenInpainter for raindrops: that checkpoint
    was trained with an inverted mask (see the comment in the package's
    UnReflect_Model_TokenInpainter.forward), so at each hole it receives the
    hole's own token and learned to correct a highlight in place. On a droplet
    it keeps the droplet and just darkens it -- the dark blobs. Passing the
    mask the documented way round is worse still (blocky grey fills), since
    the weights never learned it.
    """
    B, N, C = T.shape
    x = T.transpose(1, 2).reshape(B, C, grid, grid).clone()
    hole_g = hole.reshape(B, 1, grid, grid)
    known = (~hole_g).float()
    if known.sum() == 0:
        return T
    k = torch.ones(1, 1, 3, 3, dtype=T.dtype, device=T.device)
    while known.min() < 1:
        num = F.conv2d(x * known, k.expand(C, 1, 3, 3), padding=1, groups=C)
        den = F.conv2d(known, k, padding=1)
        new = (den > 0) & (known == 0)
        x = torch.where(new, num / den.clamp_min(1e-6), x)
        known = torch.maximum(known, new.float())
    for _ in range(smooth_passes):
        sm = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)
        x = torch.where(hole_g, sm, x)
    return x.reshape(B, C, N).transpose(1, 2)


LAMA_WEIGHTS = Path(os.environ.get("LAMA_WEIGHTS", RAINDROP_ROOT / "lama" / "big-lama.pt"))
_lama = None


def raindrop_mask(img: Image.Image, rain_threshold: float = 0.2, grow: int = 9) -> torch.Tensor:
    """DeRaindrop's raindrop map thresholded and grown a little (droplet edges
    refract too but score lower than their cores), the same rule run_stages
    uses with detector="deraindrop". [1,1,H,W] bool at the input's resolution."""
    import derainnet

    att = derainnet.attention_map(img)
    return F.max_pool2d((att > rain_threshold).float(), kernel_size=grow, stride=1, padding=grow // 2) > 0.5


@torch.no_grad()
def lama_fill(img: Image.Image, mask: torch.Tensor) -> Image.Image:
    """Inpaints the masked pixels with LaMa (big-lama, Suvorov et al., WACV 2022)
    at the image's full resolution, in pixel space, and keeps the original
    pixels everywhere else. mask: [1,1,H,W] bool, True = fill."""
    global _lama
    if _lama is None:
        _lama = torch.jit.load(str(LAMA_WEIGHTS), map_location=DEVICE).eval()
    x = TF.to_tensor(img.convert("RGB")).unsqueeze(0)
    m = mask.float().cpu()
    h, w = x.shape[-2:]
    ph, pw = -h % 8, -w % 8  # LaMa needs sides divisible by 8
    xp = F.pad(x, (0, pw, 0, ph), mode="reflect").to(DEVICE)
    mp = F.pad(m, (0, pw, 0, ph), mode="reflect").to(DEVICE)
    out = _lama(xp, (mp > 0.5).float())[..., :h, :w].clamp(0, 1).cpu()
    return TF.to_pil_image(torch.where(m > 0.5, out, x)[0])


def run_lama_stages(img: Image.Image, mask_override: torch.Tensor | None = None, rain_threshold: float = 0.2):
    """run_stages' counterpart for the LaMa fill, yielding the same
    (stage_index, seconds, image, note) tuples so the demo can show it the same
    way. LaMa works on pixels, so there is no DINOv3 encoding or decoding."""
    yield 0, 0.0, img.convert("RGB"), "Skipped: LaMa fills in pixels at full resolution, so no DINOv3 features are needed"
    t = time.perf_counter()
    if mask_override is None:
        import derainnet

        att = derainnet.attention_map(img)
        yield 1, time.perf_counter() - t, _heatmap(att), (
            f"DeRaindrop's raindrop detector: likelihood of 'raindrop here'. Peak {att.max().item():.2f}, mean {att.mean().item():.3f}"
        )
        t = time.perf_counter()
        mask = F.max_pool2d((att > rain_threshold).float(), kernel_size=9, stride=1, padding=4) > 0.5
        source = "raindrop detector"
    else:
        yield 1, 0.0, None, "Skipped: using the correct droplet mask"
        mask = mask_override
        source = "ground-truth droplet mask"
    x = TF.to_tensor(img.convert("RGB")).unsqueeze(0)
    yield 2, time.perf_counter() - t, _overlay(x, mask), (
        f"From the {source}: {100 * mask.float().mean().item():.1f}% of pixels marked for repair, in red"
    )
    t = time.perf_counter()
    out = lama_fill(img, mask)
    yield 3, time.perf_counter() - t, out, "Marked pixels inpainted by LaMa on the " + DEVICE.type.upper()
    yield 4, 0.0, out, "Skipped: LaMa already works at full resolution; original pixels kept outside the mask"


def run_stages(img: Image.Image, threshold: float = 0.3, mask_override: torch.Tensor | None = None,
               composite: bool = False, detector: str = "highlight", rain_threshold: float = 0.2,
               fill: str = "learned"):
    """Generator. Yields (stage_index, seconds, image, note) as each stage
    finishes; stage_index -1 is model loading.

    mask_override: [1,1,H,W] bool at the input's resolution; replaces the
    model's own detector when choosing what to repair.
    detector: "highlight" is the model's own (specular) detector; "deraindrop"
    swaps in DeRaindrop's raindrop attention map (derainnet.py), thresholded at
    rain_threshold, so every rainy patch is marked for repair.
    fill: "learned" is the released TokenInpainter (matches `ura inference`);
    "neighbours" rebuilds each hole from the clean tokens around it
    (neighbour_fill), which removes droplets instead of darkening them.
    composite: keep the original pixels everywhere outside the repair mask
    (blended at the input's full resolution), instead of the decoder's
    re-rendering of the whole image, which softens detail everywhere.
    """
    if _wrapper is None:
        t = time.perf_counter()
        _load()
        yield -1, time.perf_counter() - t, None, "Model loaded."

    inner = _wrapper._model
    side = _wrapper.image_size
    orig_w, orig_h = img.size
    x = TF.resize(TF.to_tensor(img.convert("RGB")).unsqueeze(0), [side, side], antialias=True).to(DEVICE)
    patch = inner.patch_size

    with torch.no_grad():
        t = time.perf_counter()
        tokens = inner.dinov3(inner.dinov3.preprocess_image(x))["selected_hidden_states"]
        n_tok, dim = tokens[0].shape[1], tokens[0].shape[2]
        grid = int(n_tok**0.5)
        yield 0, time.perf_counter() - t, TF.to_pil_image(x[0].cpu()), (
            f"{len(tokens)} feature layers, each {grid}×{grid} = {n_tok:,} patches of {dim:,} numbers"
        )

        t = time.perf_counter()
        if detector == "deraindrop":
            import derainnet

            hl = F.interpolate(derainnet.attention_map(img).to(DEVICE), size=x.shape[-2:], mode="bilinear", align_corners=False)
            what = "DeRaindrop's raindrop detector: likelihood of 'raindrop here'"
        else:
            hl = inner.decoders["highlight"](tokens)
            what = "The model's highlight detector: likelihood of 'needs removing'"
        yield 1, time.perf_counter() - t, _heatmap(hl), (
            f"{what} (brighter = more likely; square-root scaled so faint "
            f"detections show). Peak {hl.max().item():.2f}, mean {hl.mean().item():.3f}"
        )

        t = time.perf_counter()
        if detector == "deraindrop" and mask_override is None:
            # Droplet edges refract too but score lower than their cores, so grow the mask a little.
            pixel_mask = F.max_pool2d((hl > rain_threshold).float(), kernel_size=9, stride=1, padding=4) > 0.5
            source = "raindrop detector (every rainy patch)"
        elif mask_override is not None:
            pixel_mask = F.interpolate(mask_override.float().to(DEVICE), size=x.shape[-2:], mode="nearest")
            pixel_mask = F.max_pool2d(pixel_mask, kernel_size=5, stride=1, padding=2) > 0.5
            source = "ground-truth droplet mask"
        else:
            dil = max(1, x.shape[-1] // (patch * 4))
            dil = dil + 1 if dil % 2 == 0 else dil
            pixel_mask = F.max_pool2d(hl.float(), kernel_size=dil, stride=1, padding=dil // 2) > threshold
            source = "model's own detector"
        soft = "soft" in inner.token_inpainter_class_name.lower()
        patch_mask = _M.pixel_mask_to_patch_mask(pixel_mask, patch_size=patch, threshold=0, invert=False, soft=soft)
        if soft:
            patch_mask = _M.feather_token_mask(patch_mask, radius_tokens=1, smoothstep=True)
        n_marked = int((patch_mask.float() > 0.5).sum().item())
        hl_overlay_mask = F.interpolate(pixel_mask.float(), size=x.shape[-2:], mode="nearest")
        yield 2, time.perf_counter() - t, _overlay(x, hl_overlay_mask), (
            f"From the {source}: {100 * pixel_mask.float().mean().item():.1f}% of pixels, "
            f"{n_marked:,} of {n_tok:,} patches ({100 * n_marked / n_tok:.1f}%) marked for repair, in red"
        )

        t = time.perf_counter()
        is_soft = patch_mask.dtype.is_floating_point
        completed = []
        for T in tokens:
            if fill == "neighbours":
                T_inp = neighbour_fill(T, patch_mask > 0.5 if is_soft else patch_mask, grid)
            else:
                visibility = 1.0 - patch_mask if is_soft else torch.logical_not(patch_mask)
                T_inp = inner.token_inpaint(T, visibility)
            if is_soft:
                pm = patch_mask.unsqueeze(-1)
                T_comp = pm * T_inp + (1.0 - pm) * T
            else:
                T_comp = torch.where(patch_mask.unsqueeze(-1), T_inp, T)
            completed.append(T_comp)
        change = (completed[-1] - tokens[-1]).norm(dim=-1).reshape(1, 1, grid, grid)
        change = change / change.max().clamp_min(1e-8)
        change_img = F.interpolate(change, size=x.shape[-2:], mode="nearest")
        how = ("rebuilt from the clean patches around them" if fill == "neighbours"
               else "by the released inpainter")
        yield 3, time.perf_counter() - t, _overlay(x, change_img, color=(0.1, 0.8, 1.0), alpha=0.7), (
            f"Marked patches {how}. Brighter cyan = features rewritten more"
        )

        t = time.perf_counter()
        out = None
        for name, dec in inner.decoders.items():
            if name != "highlight":
                out = dec(completed).clamp(0, 1)
        out = TF.resize(out, [orig_h, orig_w], antialias=True).cpu()
        note = "Final output, resized back to the original size"
        if composite:
            m = F.interpolate(pixel_mask.float().cpu(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            m = F.avg_pool2d(m, 5, 1, 2).clamp(0, 1)
            original = TF.to_tensor(img.convert("RGB")).unsqueeze(0)
            out = (out * m + original * (1 - m)).clamp(0, 1)
            note += "; original pixels kept outside the repaired regions"
        yield 4, time.perf_counter() - t, TF.to_pil_image(out[0]), note
