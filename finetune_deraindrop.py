"""
Fine-tune DeRaindrop (derainnet.py) on VRGS rain: the first test of whether
VRGS supervision moves a raindrop model toward dashcam/CCTV conditions it never
saw -- sliding streaks, glare around light sources, wet-glass haze, driving
scenes.

DeRaindrop was trained on one rig (a glass pane in front of a camera), so it
removes round droplets but has no notion of streaks or glare. Each batch mixes
VRGS-rendered KITTI crops with DeRaindrop's own real training pairs, so it
picks up the new degradations without forgetting real droplets.

Losses follow the paper's, minus the adversarial and VGG terms (too slow on
CPU): L1 on the restored image, plus its attention loss -- each of the
recurrent attention maps regressed to the degradation mask, later iterations
weighted more (theta=0.8). VRGS provides that mask exactly; for the real pairs
it's the thresholded rain/clean difference, as in the paper.

KITTI frames 0-78 are rendered for training and 88-106 held out for testing:
they're consecutive frames from one drive, so neighbours are near-duplicates.

    python finetune_deraindrop.py --minutes 20
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

import derainnet
import metrics
from raindrop_render import RaindropRender

ROOT = Path(r"C:\Users\sachd\Downloads\raindrop_project")
KITTI = ROOT / "organized" / "KITTI_Clean" / "all" / "rgb"
REAL_TRAIN = ROOT / "extracted" / "DeRaindrop" / "train"
VRGS_TRAIN, VRGS_TEST = ROOT / "vrgs_train", ROOT / "vrgs_test"
# Every other frame: VRGS's per-droplet rendering costs several seconds a frame on CPU.
TRAIN_FRAMES, TEST_FRAMES = range(0, 80, 2), range(88, 108, 2)
WIDTH = 960  # the demo's frame size; VRGS scales droplet sizes with the frame
CROP = 256
RESULTS = Path(__file__).parent / "eval_results" / "deraindrop_vrgs_finetune.json"


def _fit(img: Image.Image) -> Image.Image:
    s = WIDTH / img.width
    return img.resize((WIDTH, round(img.height * s)), Image.BICUBIC)


def render_pairs(frames, out_dir: Path, variants: int, seed: int):
    """Writes {stem}_{k}_rainy.png, _clean.png and _mask.png; skips if already there."""
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = RaindropRender(height=WIDTH, width=WIDTH)
    for f in frames:
        clean = _fit(Image.open(KITTI / f"{f:010d}.png").convert("RGB"))
        for k in range(variants):
            stem = f"{f:010d}_{k}"
            if (out_dir / f"{stem}_mask.png").exists():
                continue
            random.seed(seed + 1000 * f + k)
            torch.manual_seed(seed + 1000 * f + k)
            intensity = random.uniform(0.3, 0.9)
            context = random.choice(["dashcam", "cctv"])
            with torch.no_grad():
                r = renderer(TF.to_tensor(clean).unsqueeze(0), rain_intensity=intensity, capture_context=context)
            TF.to_pil_image(r["degraded"][0]).save(out_dir / f"{stem}_rainy.png")
            clean.save(out_dir / f"{stem}_clean.png")
            TF.to_pil_image(r["mask"][0]).save(out_dir / f"{stem}_mask.png")


# Pairs are kept as paths and cropped on load: all 861 real pairs as float
# tensors would need ~8 GB.
def _vrgs_paths(d: Path):
    return [tuple(d / f"{m.name[: -len('_mask.png')]}_{s}.png" for s in ("rainy", "clean", "mask"))
            for m in sorted(d.glob("*_mask.png"))]


def _real_paths():
    pairs = []
    for rain in sorted((REAL_TRAIN / "data").glob("*_rain.*")):
        gt = REAL_TRAIN / "gt" / rain.name.replace("_rain", "_clean")
        if gt.exists():
            pairs.append((rain, gt, None))
    return pairs


def _crop(paths):
    rain_p, clean_p, mask_p = paths
    r, g = Image.open(rain_p).convert("RGB"), Image.open(clean_p).convert("RGB")
    w, h = min(r.width, g.width), min(r.height, g.height)
    x, y = random.randint(0, w - CROP), random.randint(0, h - CROP)
    box = (x, y, x + CROP, y + CROP)
    r, g = TF.to_tensor(r.crop(box)), TF.to_tensor(g.crop(box))
    if mask_p is not None:
        m = TF.to_tensor(Image.open(mask_p).crop(box))
    else:  # the paper's attention target: pixels the rain changed by more than 30/255
        m = ((r - g).abs().max(0, keepdim=True).values > 30 / 255).float()
    out = [r, g, m]
    if random.random() < 0.5:
        out = [t.flip(-1) for t in out]
    return out


def _bgr(t):  # the network works in BGR, as it was trained with OpenCV
    return t[:, [2, 1, 0]]


def train(minutes: float, batch: int, lr: float, real_frac: float):
    vrgs, real = _vrgs_paths(VRGS_TRAIN), _real_paths()
    print(f"{len(vrgs)} VRGS training pairs, {len(real)} real DeRaindrop pairs", flush=True)
    model = derainnet.Generator()
    model.load_state_dict(torch.load(derainnet.WEIGHTS, map_location="cpu"))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    deadline, step = time.time() + minutes * 60, 0
    while time.time() < deadline:
        crops = [_crop(random.choice(real if random.random() < real_frac else vrgs)) for _ in range(batch)]
        rainy, clean, mask = (torch.stack(c) for c in zip(*crops))
        masks, out = model(_bgr(rainy))
        loss_att = sum(0.8 ** (len(masks) - 1 - i) * F.mse_loss(a, mask) for i, a in enumerate(masks))
        loss_img = F.l1_loss(out, _bgr(clean))
        loss = loss_img + 0.1 * loss_att
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 10 == 0:
            print(f"step {step}: L1 {loss_img.item():.4f}  attention {loss_att.item():.4f}  "
                  f"({(deadline - time.time()) / 60:.1f} min left)", flush=True)
    derainnet.FINETUNED.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), derainnet.FINETUNED)
    print(f"Saved {derainnet.FINETUNED} after {step} steps", flush=True)


def _scores(p: Image.Image, g: Image.Image):
    p = TF.to_tensor(p.convert("RGB")).unsqueeze(0)
    g = TF.to_tensor(g.convert("RGB").resize((p.shape[-1], p.shape[-2]))).unsqueeze(0)
    return metrics.psnr_metric(p, g).item(), metrics.ssim_metric(p, g).item()


def evaluate(name: str, pairs) -> dict:
    """pairs: list of (rainy, clean) PIL images. Prints and returns input /
    released / fine-tuned averages."""
    rows = []
    for rainy, clean in pairs:
        rows.append((_scores(rainy, clean),
                     _scores(derainnet.derain(rainy)[0], clean),
                     _scores(derainnet.derain(rainy, derainnet.FINETUNED)[0], clean)))
    n = len(rows)
    result = {"images": n}
    for i, label in enumerate(["rainy input", "DeRaindrop (released)", "DeRaindrop + VRGS fine-tune"]):
        p, s = sum(r[i][0] for r in rows) / n, sum(r[i][1] for r in rows) / n
        result[label] = {"psnr": round(p, 2), "ssim": round(s, 3)}
        beats = ""
        if i > 0:
            result[label]["beats_input"] = sum(r[i][0] > r[0][0] and r[i][1] > r[0][1] for r in rows)
            beats = f"  beats input on both: {result[label]['beats_input']}/{n}"
        print(f"{name} | {label:30s} {p:.2f} dB / SSIM {s:.3f}{beats}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    t = time.time()
    render_pairs(TRAIN_FRAMES, VRGS_TRAIN, variants=2, seed=0)
    render_pairs(TEST_FRAMES, VRGS_TEST, variants=1, seed=999)
    print(f"VRGS pairs ready in {time.time() - t:.0f}s", flush=True)
    if not args.eval_only:
        train(args.minutes, args.batch, args.lr, args.real_frac)

    results = {"vrgs_heldout": evaluate(
        "VRGS held-out", [(Image.open(VRGS_TEST / f"{m.name[:-9]}_rainy.png"), Image.open(VRGS_TEST / f"{m.name[:-9]}_clean.png"))
                          for m in sorted(VRGS_TEST.glob("*_mask.png"))])}
    test_a = ROOT / "extracted" / "DeRaindrop" / "test_a"
    # Every third image keeps the CPU evaluation to a few minutes.
    real = [(Image.open(p), Image.open(test_a / "gt" / p.name.replace("_rain", "_clean")))
            for p in sorted((test_a / "data").glob("*_rain.*"), key=lambda p: int(p.name.split("_")[0]))[::3]]
    results["real_test_a"] = evaluate(f"Real (DeRaindrop test_a, {len(real)} images)", real)
    RESULTS.parent.mkdir(exist_ok=True)
    results["run"] = f"{args.minutes:g} min, batch {args.batch}, lr {args.lr:g}, real fraction {args.real_frac:g}"
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"Wrote {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
