"""
Scores every method on the same test sets, for one comparison table
(eval_results/all_methods.json):

  * real rain: all DeRaindrop test_a photos (never trained on), and
  * VRGS dashcam/CCTV rain on a held-out KITTI drive (finetune_deraindrop.py's
    vrgs2_test renders of drive_0113, never trained on).

Methods: the rainy input left alone; stock UnReflectAnything; the raindrop
detector + neighbour fill; the raindrop detector + LaMa; DeRaindrop released and
fine-tuned on VRGS; and, if its weights exist, our VRGS-trained model: its
raindrop detector with LaMa, and with its own (one-hour) token inpainter. PSNR /
SSIM against the clean image; "beats the input" means better on both. Also
writes ours (detector + LaMa) for the demo's examples to
DEMO_ASSETS/trained_output.

    python eval_all.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

import derainnet
import metrics
import model_stages as ms
from raindrop_paths import RAINDROP_ROOT

ASSETS = Path(os.environ.get("DEMO_ASSETS", RAINDROP_ROOT / "demo_assets"))
TEST_A = RAINDROP_ROOT / "extracted" / "DeRaindrop" / "test_a"
VRGS_TEST = RAINDROP_ROOT / "vrgs2_test"
RESULTS = Path(__file__).parent / "eval_results" / "all_methods.json"


def _final(stages) -> Image.Image:
    out = None
    for _, _, im, _ in stages:
        out = im if im is not None else out
    return out


METHODS = {
    "UnReflectAnything, stock": lambda im: _final(ms.run_stages(im)),
    "Raindrop detector + neighbour fill": lambda im: _final(
        ms.run_stages(im, detector="deraindrop", fill="neighbours", composite=True)),
    "Raindrop detector + LaMa": lambda im: ms.lama_fill(im, ms.raindrop_mask(im)),
    "DeRaindrop (released)": lambda im: derainnet.derain(im)[0],
    "DeRaindrop + VRGS fine-tune": lambda im: derainnet.derain(im, derainnet.FINETUNED)[0],
    "Ours: VRGS-trained detector + LaMa": lambda im: ms.lama_fill(im, ms.trained_raindrop_mask(im)),
    "Ours: VRGS-trained detector + its own inpainter": lambda im: _final(
        ms.run_stages(im, composite=True, trained=True, threshold=ms.TRAINED_THRESHOLD)),
}
DEMO_METHOD = "Ours: VRGS-trained detector + LaMa"  # what the demo shows as ours


def _scores(p: Image.Image, g: Image.Image):
    p = TF.to_tensor(p.convert("RGB")).unsqueeze(0)
    g = TF.to_tensor(g.convert("RGB").resize((p.shape[-1], p.shape[-2]))).unsqueeze(0)
    return metrics.psnr_metric(p, g).item(), metrics.ssim_metric(p, g).item()


def evaluate(name: str, pairs, methods: dict) -> dict:
    rows = {k: [] for k in ["rainy input", *methods]}
    for rainy, clean in pairs:
        rows["rainy input"].append(_scores(rainy, clean))
        for label, fn in methods.items():
            rows[label].append(_scores(fn(rainy), clean))
    n = len(pairs)
    result = {"images": n}
    for label, r in rows.items():
        s = {"psnr": round(sum(x[0] for x in r) / n, 2), "ssim": round(sum(x[1] for x in r) / n, 3)}
        if label != "rainy input":
            s["beats_input"] = sum(a[0] > b[0] and a[1] > b[1] for a, b in zip(r, rows["rainy input"]))
        result[label] = s
        print(f"{name} | {label:42s} {s['psnr']:.2f} dB / SSIM {s['ssim']:.3f}"
              + (f"  beats input on both: {s['beats_input']}/{n}" if "beats_input" in s else ""), flush=True)
    return result


def main():
    methods = {k: v for k, v in METHODS.items() if "Ours" not in k or ms.TRAINED_WEIGHTS.exists()}
    t = time.time()
    real = [(Image.open(p).convert("RGB"), Image.open(TEST_A / "gt" / p.name.replace("_rain", "_clean")).convert("RGB"))
            for p in sorted((TEST_A / "data").glob("*_rain.*"), key=lambda p: int(p.name.split("_")[0]))]
    vrgs = [(Image.open(m.with_name(m.name.replace("_mask", "_rainy"))).convert("RGB"),
             Image.open(m.with_name(m.name.replace("_mask", "_clean"))).convert("RGB"))
            for m in sorted(VRGS_TEST.glob("*_mask.png"))]
    results = {
        "real_test_a": evaluate(f"Real (DeRaindrop test_a, {len(real)})", real, methods),
        "vrgs_heldout": evaluate(f"VRGS held-out drive ({len(vrgs)})", vrgs, methods),
        "run": f"{ms.DEVICE.type.upper()}; our model: {ms.TRAINED_WEIGHTS.name if ms.TRAINED_WEIGHTS.exists() else 'not trained yet'}",
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {RESULTS} in {time.time() - t:.0f}s", flush=True)

    if ms.TRAINED_WEIGHTS.exists():
        out_dir = ASSETS / "trained_output"
        out_dir.mkdir(exist_ok=True)
        for p in (ASSETS / "rainy").glob("*.png"):
            METHODS[DEMO_METHOD](Image.open(p).convert("RGB")).save(out_dir / p.name)
        print(f"Wrote our outputs for the demo examples to {out_dir}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
