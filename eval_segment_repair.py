"""
Scores the training-free segment-and-repair pipeline (segment_repair.py) on the
58 real rain/clean pairs of DeRaindrop's test_a, the same photos, resolution and
PSNR/SSIM code as finetune_deraindrop.py, so the numbers compare directly with
eval_results/deraindrop_vrgs_finetune.json. Four outputs per photo:

  rainy input
  ours, whole frame               our DeRaindrop fine-tuned on VRGS, applied everywhere
  segment & repair, instances     only the found instances repaired (repair(whole_frame=False))
  segment & repair, hybrid        whole-frame restore + the instance rules (repair(), the default)
  hybrid + SAM 3                  the hybrid with SAM 3's wet road and lens glare

An earlier run, before the hybrid existed, is in
eval_results/segment_repair_real_test_a_instances_only.json (instances only,
with and without SAM 3: both 28.06 dB / SSIM 0.888).

The heavy models (Flare7K++, DeRaindrop, YOLO11-seg, SAM 3) run once per photo;
both pipeline variants reuse their outputs, since only the instance split differs.
Writes eval_results/segment_repair_real_test_a.json (averages + every photo).

    python eval_segment_repair.py [--no-sam3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from PIL import Image
from torchvision.transforms import functional as TF

import derainnet
import metrics
import segment_repair as sr
from raindrop_paths import RAINDROP_ROOT

TEST_A = RAINDROP_ROOT / "extracted" / "DeRaindrop" / "test_a"
RESULTS = Path(__file__).parent / "eval_results" / "segment_repair_real_test_a.json"


def _scores(p: Image.Image, g: Image.Image):
    p = TF.to_tensor(p.convert("RGB")).unsqueeze(0)
    g = TF.to_tensor(g.convert("RGB").resize((p.shape[-1], p.shape[-2]))).unsqueeze(0)
    return metrics.psnr_metric(p, g).item(), metrics.ssim_metric(p, g).item()


def _cache_models():
    """Memoise the four models by image content, so a second analyse() of the
    same photo only redoes the instance split."""
    cache: dict = {}
    for mod, name in ((sr, "flare7kpp"), (sr, "detect_objects"), (sr, "find_concepts"), (derainnet, "derain")):
        fn = getattr(mod, name)

        def wrapped(img, *args, _fn=fn, _name=name, **kw):
            key = (_name, hashlib.md5(img.convert("RGB").tobytes()).hexdigest(), args, tuple(sorted(kw.items())))
            if key not in cache:
                cache[key] = _fn(img, *args, **kw)
            return cache[key]
        setattr(mod, name, wrapped)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sam3", action="store_true")
    args = ap.parse_args()
    use_sam3 = not args.no_sam3 and sr.SAM3_WEIGHTS.exists()
    cache = _cache_models()
    pairs = sorted((TEST_A / "data").glob("*_rain.*"), key=lambda p: int(p.name.split("_")[0]))
    labels = (["rainy input", "ours, whole frame", "segment & repair, instances", "segment & repair, hybrid"]
              + (["hybrid + SAM 3"] if use_sam3 else []))
    per_photo = []
    t0 = time.perf_counter()
    for k, p in enumerate(pairs, 1):
        rainy = Image.open(p).convert("RGB")
        clean = Image.open(TEST_A / "gt" / p.name.replace("_rain", "_clean"))
        outs = [rainy]
        a = sr.analyse(rainy)
        outs += [a.derained, sr.repair(a, whole_frame=False), sr.repair(a)]
        if use_sam3:
            outs.append(sr.repair(sr.analyse(rainy, use_sam3=True)))
        cache.clear()
        row = {"photo": p.name}
        for label, im in zip(labels, outs):
            ps, ss = _scores(im, clean)
            row[label] = {"psnr": round(ps, 3), "ssim": round(ss, 4)}
        per_photo.append(row)
        print(f"{k}/{len(pairs)} {p.name} {time.perf_counter() - t0:.0f}s | "
              + " | ".join(f"{lab}: {row[lab]['psnr']:.2f}/{row[lab]['ssim']:.3f}" for lab in labels), flush=True)

    n = len(per_photo)
    summary = {"images": n}
    for label in labels:
        ps = sum(r[label]["psnr"] for r in per_photo) / n
        ss = sum(r[label]["ssim"] for r in per_photo) / n
        summary[label] = {"psnr": round(ps, 2), "ssim": round(ss, 3)}
        if label != "rainy input":
            summary[label]["beats_input"] = sum(r[label]["psnr"] > r["rainy input"]["psnr"]
                                                and r[label]["ssim"] > r["rainy input"]["ssim"] for r in per_photo)
        if label not in ("rainy input", "ours, whole frame"):
            base = "ours, whole frame"
            summary[label]["beats_whole_frame_psnr"] = sum(r[label]["psnr"] > r[base]["psnr"] for r in per_photo)
            summary[label]["beats_whole_frame_ssim"] = sum(r[label]["ssim"] > r[base]["ssim"] for r in per_photo)
        print(f"{label:28s} {summary[label]}", flush=True)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"real_test_a": summary, "photos": per_photo}, indent=1) + "\n")
    print(f"wrote {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
