"""
Pre-renders a cached VRGS training set, so training isn't waiting on the
renderer (it is launch-bound on the GPU, ~0.3 s per 448 px frame, which would
take most of each training step).

Each clean KITTI frame is rendered `--variants` times, each with its own square
crop, flip, rain intensity and camera context, at the training resolution.
The result is one file of uint8 tensors {clean, rainy, mask} that
RaindropEngine draws training batches from when the config sets VRGS_CACHE.
Validation keeps rendering on the fly (with a fixed seed), so it is unaffected.

    python prerender_vrgs.py --size 448 --variants 4

The held-out scene (drive_0113) is never rendered here.
"""
from __future__ import annotations

import argparse
import random
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from raindrop_paths import RAINDROP_ROOT
from raindrop_render import RaindropRender

CLEAN = RAINDROP_ROOT / "organized" / "KITTI_Clean"
HELD_OUT = {"drive_0113"}

_renderer = None


def _render(job):
    """Worker: one (clean, rainy, mask) triplet as uint8 tensors."""
    global _renderer
    path, k, size = job
    if _renderer is None:
        torch.set_num_threads(1)
        random.seed(0)
        _renderer = RaindropRender(height=size, width=size)
    seed = zlib.crc32(f"{path}|{k}".encode())
    random.seed(seed)
    torch.manual_seed(seed)
    img = Image.open(path).convert("RGB")
    s = size / min(img.size)  # resize the short side to `size`, then a random square crop
    img = img.resize((max(size, round(img.width * s)), max(size, round(img.height * s))), Image.BICUBIC)
    x, y = random.randint(0, img.width - size), random.randint(0, img.height - size)
    clean = TF.to_tensor(img.crop((x, y, x + size, y + size)))
    if random.random() < 0.5:
        clean = clean.flip(-1)
    context = "dashcam" if random.random() < 0.7 else "cctv"
    with torch.no_grad():
        r = _renderer(clean.unsqueeze(0), rain_intensity=random.uniform(0.2, 1.0), capture_context=context)
    to8 = lambda t: (t.clamp(0, 1) * 255 + 0.5).to(torch.uint8)  # noqa: E731
    return to8(clean), to8(r["degraded"][0]), to8(r["mask"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or RAINDROP_ROOT / f"vrgs_cache_{args.size}.pt"

    scenes = [s for s in sorted(CLEAN.iterdir()) if s.is_dir() and s.name not in HELD_OUT]
    frames = [p for s in scenes for p in sorted((s / "rgb").glob("*.png"))]
    jobs = [(p, k, args.size) for p in frames for k in range(args.variants)]
    print(f"{len(frames)} frames from {[s.name for s in scenes]} x {args.variants} variants", flush=True)
    t = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        triplets = list(ex.map(_render, jobs, chunksize=8))
    cache = {k: torch.stack([tr[i] for tr in triplets]) for i, k in enumerate(("clean", "rainy", "mask"))}
    cache["scenes"] = [s.name for s in scenes]
    torch.save(cache, out)
    mb = sum(v.numel() for k, v in cache.items() if k != "scenes") / 2**20
    print(f"{len(jobs)} triplets in {time.time() - t:.0f}s -> {out} ({mb:.0f} MB); "
          f"mask covers {100 * (cache['mask'] > 25).float().mean():.1f}% of pixels", flush=True)


if __name__ == "__main__":
    main()
