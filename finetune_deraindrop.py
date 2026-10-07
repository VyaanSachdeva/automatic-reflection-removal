"""
Fine-tune DeRaindrop (derainnet.py) on VRGS rain: the first test of whether
VRGS supervision moves a raindrop model toward dashcam/CCTV conditions it never
saw -- sliding streaks, glare around light sources, wet-glass haze, driving
scenes.

DeRaindrop was trained on one rig (a glass pane in front of a camera), so it
removes round droplets but has no notion of streaks or glare. Each batch mixes
VRGS-rendered KITTI crops with DeRaindrop's own real training pairs, so it
picks up the new degradations without forgetting real droplets.

Losses follow the paper's, minus the adversarial and VGG terms: L1 on the
restored image, plus its attention loss -- each of the recurrent attention
maps regressed to the degradation mask, later iterations weighted more
(theta=0.8). VRGS provides that mask exactly; for the real pairs it's the
thresholded rain/clean difference, as in the paper.

Every KITTI scene in organized/KITTI_Clean is rendered for training except
TEST_SCENE, a whole drive held out for testing (consecutive frames of one drive
are near-duplicates, so holding out frames of a training drive would flatter
the result). Runs on the GPU when there is one; VRGS pairs are rendered in
parallel CPU processes and cached.

    python finetune_deraindrop.py --minutes 30
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

import derainnet
import metrics
from raindrop_paths import RAINDROP_ROOT
from raindrop_render import RaindropRender

ROOT = RAINDROP_ROOT
CLEAN = ROOT / "organized" / "KITTI_Clean"
TEST_SCENE = "drive_0113"  # a whole drive never trained on
REAL_TRAIN = ROOT / "extracted" / "DeRaindrop" / "train"
VRGS_TRAIN, VRGS_TEST = ROOT / "vrgs2_train", ROOT / "vrgs2_test"  # rendered with the current VRGS
PREVIOUS = ROOT / "checkpoints" / "deraindrop_vrgs_cpu20min.pkl"  # the first, 20-min CPU fine-tune
WIDTH = 960  # the demo's frame size; VRGS scales droplet sizes with the frame
CROP = 256
RESULTS = Path(__file__).parent / "eval_results" / "deraindrop_vrgs_finetune.json"
LABELS = ["rainy input", "DeRaindrop (released)", "DeRaindrop + VRGS fine-tune (previous: CPU, 20 min)",
          "DeRaindrop + VRGS fine-tune"]


def _fit(img: Image.Image) -> Image.Image:
    s = WIDTH / img.width
    return img.resize((WIDTH, round(img.height * s)), Image.BICUBIC)


def _frames(test: bool) -> list[Path]:
    scenes = [CLEAN / TEST_SCENE] if test else [s for s in sorted(CLEAN.iterdir()) if s.is_dir() and s.name != TEST_SCENE]
    return [p for s in scenes for p in sorted((s / "rgb").glob("*.png"))]


_renderer = None


def _render_one(job):
    """Worker: renders one VRGS pair. Writes {stem}_rainy.png, _clean.png and _mask.png."""
    global _renderer
    path, stem, out_dir = job
    if (out_dir / f"{stem}_mask.png").exists():
        return
    if _renderer is None:
        torch.set_num_threads(1)
        random.seed(0)
        _renderer = RaindropRender(height=WIDTH, width=WIDTH)
    clean = _fit(Image.open(path).convert("RGB"))
    seed = zlib.crc32(stem.encode())
    random.seed(seed)
    torch.manual_seed(seed)
    intensity = random.uniform(0.3, 0.9)
    context = random.choice(["dashcam", "cctv"])
    with torch.no_grad():
        r = _renderer(TF.to_tensor(clean).unsqueeze(0), rain_intensity=intensity, capture_context=context)
    TF.to_pil_image(r["degraded"][0]).save(out_dir / f"{stem}_rainy.png")
    clean.save(out_dir / f"{stem}_clean.png")
    TF.to_pil_image(r["mask"][0]).save(out_dir / f"{stem}_mask.png")


def render_pairs(frames: list[Path], out_dir: Path, variants: int, workers: int):
    """Renders each frame `variants` times (skipping pairs already there), in parallel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, f"{p.parent.parent.name}_{p.stem}_{k}", out_dir) for p in frames for k in range(variants)]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_render_one, jobs, chunksize=8))


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


class _Crops(torch.utils.data.IterableDataset):
    """Endless random crops, a real_frac share of them from real pairs."""

    def __init__(self, vrgs, real, real_frac: float):
        self.vrgs, self.real, self.real_frac = vrgs, real, real_frac

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        random.seed(time.time_ns() + (info.id if info else 0))
        while True:
            yield _crop(random.choice(self.real if random.random() < self.real_frac else self.vrgs))


def _bgr(t):  # the network works in BGR, as it was trained with OpenCV
    return t[:, [2, 1, 0]]


def train(minutes: float, batch: int, lr: float, real_frac: float, workers: int):
    vrgs, real = _vrgs_paths(VRGS_TRAIN), _real_paths()
    print(f"{len(vrgs)} VRGS training pairs, {len(real)} real DeRaindrop pairs, on {derainnet.DEVICE}", flush=True)
    if derainnet.FINETUNED.exists() and not PREVIOUS.exists():
        shutil.copy2(derainnet.FINETUNED, PREVIOUS)  # keep the first fine-tune for comparison
    model = derainnet.Generator()
    model.load_state_dict(torch.load(derainnet.WEIGHTS, map_location="cpu"))
    model.to(derainnet.DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(_Crops(vrgs, real, real_frac), batch_size=batch, num_workers=workers,
                                         pin_memory=derainnet.DEVICE.type == "cuda")
    derainnet.FINETUNED.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    deadline, next_save, step = start + minutes * 60, start + 300, 0
    for rainy, clean, mask in loader:
        rainy, clean, mask = (t.to(derainnet.DEVICE, non_blocking=True) for t in (rainy, clean, mask))
        masks, out = model(_bgr(rainy))
        loss_att = sum(0.8 ** (len(masks) - 1 - i) * F.mse_loss(a, mask) for i, a in enumerate(masks))
        loss_img = F.l1_loss(out, _bgr(clean))
        loss = loss_img + 0.1 * loss_att
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step}: L1 {loss_img.item():.4f}  attention {loss_att.item():.4f}  "
                  f"({(deadline - time.time()) / 60:.1f} min left)", flush=True)
        if time.time() > next_save:  # a stopped run still leaves usable weights
            torch.save(model.state_dict(), derainnet.FINETUNED)
            next_save += 300
        if time.time() > deadline:
            break
    torch.save(model.state_dict(), derainnet.FINETUNED)
    print(f"Saved {derainnet.FINETUNED} after {step} steps ({step * batch} crops)", flush=True)
    return step


def _scores(p: Image.Image, g: Image.Image):
    p = TF.to_tensor(p.convert("RGB")).unsqueeze(0)
    g = TF.to_tensor(g.convert("RGB").resize((p.shape[-1], p.shape[-2]))).unsqueeze(0)
    return metrics.psnr_metric(p, g).item(), metrics.ssim_metric(p, g).item()


def evaluate(name: str, pairs) -> dict:
    """pairs: list of (rainy, clean) PIL images. Prints and returns averages for
    the input, the released model and each fine-tune."""
    weights = [None, derainnet.WEIGHTS, PREVIOUS, derainnet.FINETUNED]
    labels = [lab for lab, w in zip(LABELS, weights) if w is None or w.exists()]
    weights = [w for w in weights if w is None or w.exists()]
    rows = [[_scores(rainy if w is None else derainnet.derain(rainy, w)[0], clean) for w in weights]
            for rainy, clean in pairs]
    n = len(rows)
    result = {"images": n}
    for i, label in enumerate(labels):
        p, s = sum(r[i][0] for r in rows) / n, sum(r[i][1] for r in rows) / n
        result[label] = {"psnr": round(p, 2), "ssim": round(s, 3)}
        beats = ""
        if i > 0:
            result[label]["beats_input"] = sum(r[i][0] > r[0][0] and r[i][1] > r[0][1] for r in rows)
            beats = f"  beats input on both: {result[label]['beats_input']}/{n}"
        print(f"{name} | {label:52s} {p:.2f} dB / SSIM {s:.3f}{beats}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--variants", type=int, default=4, help="VRGS renders per training frame")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    t = time.time()
    train_frames = _frames(test=False)
    render_pairs(train_frames, VRGS_TRAIN, args.variants, args.workers)
    print(f"VRGS training pairs ready in {time.time() - t:.0f}s: {len(train_frames)} frames x {args.variants}",
          flush=True)
    steps = None
    if not args.eval_only:
        steps = train(args.minutes, args.batch, args.lr, args.real_frac, args.workers)

    # Rendered after training, so the held-out drive can still be downloading while it trains.
    test_frames = _frames(test=True)[::5]
    if not test_frames:
        raise SystemExit(f"No frames for the held-out drive {TEST_SCENE} yet: run download_kitti.py "
                         f"{TEST_SCENE[-4:]}, then this script with --eval-only")
    render_pairs(test_frames, VRGS_TEST, 1, args.workers)
    print(f"{len(test_frames)} held-out frames from {TEST_SCENE}", flush=True)

    results = {"vrgs_heldout": evaluate(
        f"VRGS held-out ({TEST_SCENE})",
        [(Image.open(VRGS_TEST / f"{m.name[:-9]}_rainy.png"), Image.open(VRGS_TEST / f"{m.name[:-9]}_clean.png"))
         for m in sorted(VRGS_TEST.glob("*_mask.png"))])}
    test_a = ROOT / "extracted" / "DeRaindrop" / "test_a"
    real = [(Image.open(p), Image.open(test_a / "gt" / p.name.replace("_rain", "_clean")))
            for p in sorted((test_a / "data").glob("*_rain.*"), key=lambda p: int(p.name.split("_")[0]))]
    results["real_test_a"] = evaluate(f"Real (DeRaindrop test_a, {len(real)} images)", real)
    RESULTS.parent.mkdir(exist_ok=True)
    results["run"] = (f"{derainnet.DEVICE.type.upper()}, {args.minutes:g} min"
                      + (f", {steps} steps" if steps else "")
                      + f" of batch {args.batch} (256px crops, {1 - args.real_frac:.0%} VRGS / {args.real_frac:.0%} real "
                        f"DeRaindrop pairs), lr {args.lr:g}; VRGS on {len(train_frames)} KITTI frames x {args.variants}, "
                        f"tested on held-out drive {TEST_SCENE}")
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
