"""
Scores LaMa fill (model_stages.lama_fill) on the demo's real DeRaindrop test
photos against neighbour fill and the untouched input, scored the same way as
the demo's comparison table (PSNR / SSIM against the clean photo; "beats the
input" means better on both). Also fills with the true droplet mask
(model_stages.ground_truth_mask, from the clean photo) as an upper bound on
what better detection could give. Writes LaMa's outputs to
DEMO_ASSETS/lama_output so the demo can show them, and the scores to
eval_results/lama_fill.json.

    python eval_lama.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

import metrics
import model_stages as ms
from raindrop_paths import RAINDROP_ROOT

ASSETS = Path(os.environ.get("DEMO_ASSETS", RAINDROP_ROOT / "demo_assets"))
RESULTS = Path(__file__).parent / "eval_results" / "lama_fill.json"


def _scores(pred: Image.Image, gt: Image.Image) -> tuple[float, float]:
    p = TF.to_tensor(pred.convert("RGB")).unsqueeze(0)
    g = TF.to_tensor(gt.convert("RGB").resize(pred.size)).unsqueeze(0)
    return metrics.psnr_metric(p, g).item(), metrics.ssim_metric(p, g).item()


def _summary(rows: list[tuple[float, float]], rainy: list[tuple[float, float]] | None) -> dict:
    n = len(rows)
    s = {"psnr": round(sum(r[0] for r in rows) / n, 2), "ssim": round(sum(r[1] for r in rows) / n, 3)}
    if rainy is not None:
        s["beats_input"] = sum(r[0] > i[0] and r[1] > i[1] for r, i in zip(rows, rainy))
    return s


def main():
    out_dir = ASSETS / "lama_output"
    out_dir.mkdir(exist_ok=True)
    paths = [p for p in sorted((ASSETS / "rainy").glob("*.png"), key=lambda p: int(p.stem))
             if (ASSETS / "gt" / p.name).exists() and (ASSETS / "drnfill_output" / p.name).exists()]
    rainy_s, nfill_s, lama_s, masked, secs = [], [], [], [], []
    gt_lama_s, gt_masked = [], []  # diagnostic upper bound: the true droplet mask instead of the detector
    for path in paths:
        rainy, gt = Image.open(path).convert("RGB"), Image.open(ASSETS / "gt" / path.name)
        if ms.DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        mask = ms.raindrop_mask(rainy)
        out = ms.lama_fill(rainy, mask)
        if ms.DEVICE.type == "cuda":
            torch.cuda.synchronize()
        secs.append(time.perf_counter() - t)
        out.save(out_dir / path.name)
        masked.append(mask.float().mean().item())
        rainy_s.append(_scores(rainy, gt))
        nfill_s.append(_scores(Image.open(ASSETS / "drnfill_output" / path.name), gt))
        lama_s.append(_scores(out, gt))
        gt_mask = ms.ground_truth_mask(rainy, gt)
        gt_masked.append(gt_mask.float().mean().item())
        gt_lama_s.append(_scores(ms.lama_fill(rainy, gt_mask), gt))
        print(f"{path.name}: rainy {rainy_s[-1][0]:.2f}/{rainy_s[-1][1]:.3f}  neighbour fill "
              f"{nfill_s[-1][0]:.2f}/{nfill_s[-1][1]:.3f}  LaMa {lama_s[-1][0]:.2f}/{lama_s[-1][1]:.3f}  "
              f"({100 * masked[-1]:.1f}% masked, {secs[-1]:.2f}s)", flush=True)

    # The first image includes loading both models and TorchScript's warm-up.
    steady = sorted(secs[1:])[len(secs[1:]) // 2] if len(secs) > 1 else secs[0]
    result = {
        "run": f"DeRaindrop's raindrop map (threshold 0.2, grown 9 px) + big-lama, full resolution, "
               f"{ms.DEVICE.type.upper()}{' ' + torch.cuda.get_device_name(0) if ms.DEVICE.type == 'cuda' else ''}",
        "real_test_a": {
            "images": len(paths),
            "rainy input": _summary(rainy_s, None),
            "Raindrop detector + neighbour fill": _summary(nfill_s, rainy_s),
            "Raindrop detector + LaMa": _summary(lama_s, rainy_s),
            "True droplet mask + LaMa (upper bound)": _summary(gt_lama_s, rainy_s),
        },
        "lama": {
            "pixels_masked_pct": round(100 * sum(masked) / len(masked), 1),
            "true_mask_pixels_pct": round(100 * sum(gt_masked) / len(gt_masked), 1),
            "latency_s_median": round(steady, 2),
        },
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
