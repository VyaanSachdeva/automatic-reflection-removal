"""
Runs real CCTV rain footage (raindrop_project/Defects(all): day, night and
ghosting scenarios, one folder per mix of defects) through our model -- DeRaindrop
fine-tuned on VRGS -- and the released DeRaindrop, and writes:

  Defects(all)_derained/<same folders>/<name>.png      ours
  Defects(all)_deraindrop_released/<same folders>/...  the released model
  Defects(all)_comparisons/<category> - <scenario>.jpg  rainy | released | ours, one row per image
  Defects(all)_comparisons/metrics.csv, summary.json    no-reference measurements

With --segment it also runs the training-free segment-and-repair pipeline
(segment_repair.py: the hybrid, with SAM 3 when its weights are there) and writes:

  Defects(all)_segment_repair/<same folders>/<name>.png            repaired frames
  Defects(all)_segment_repair_instances/<same folders>/<name>.jpg  the instances it found
  Defects(all)_comparisons_segment_repair/...   rainy | instances | released | ours | segment & repair

There is no clean ground truth for this footage, so PSNR/SSIM can't be computed.
The measurements describe what each model did instead: how much it changed the
frame (mean absolute change, 0-255), and the ratio of output to input sharpness
(variance of the Laplacian) and contrast (std of grey levels). Below 1 = softer /
flatter than the input, above 1 = sharper / more contrast.

The images are 2x screenshots, so they are processed at half size, which is about
their native resolution (~1360x760). Every run skips outputs that already exist,
so it can be stopped and resumed.

    python eval_defects.py              # both models
    python eval_defects.py --ours-only
    python eval_defects.py --segment    # + segment & repair (~70 s a frame on a laptop CPU with SAM 3)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import laplace

import derainnet
from raindrop_paths import RAINDROP_ROOT

SRC = RAINDROP_ROOT / "Defects(all)"
OURS = RAINDROP_ROOT / "Defects(all)_derained"
RELEASED = RAINDROP_ROOT / "Defects(all)_deraindrop_released"
COMP = RAINDROP_ROOT / "Defects(all)_comparisons"
SEG = RAINDROP_ROOT / "Defects(all)_segment_repair"
SEG_INST = RAINDROP_ROOT / "Defects(all)_segment_repair_instances"
SEG_COMP = RAINDROP_ROOT / "Defects(all)_comparisons_segment_repair"
MODELS = {"ours": (derainnet.FINETUNED, OURS), "released": (derainnet.WEIGHTS, RELEASED)}
TITLES = {"released": "DeRaindrop, released", "ours": "Ours: fine-tuned on VRGS",
          "segment": "New: segment & repair"}


def _load(p: Path) -> Image.Image:
    im = Image.open(p).convert("RGB")
    return im.resize((im.width // 2, im.height // 2), Image.LANCZOS)


def _stats(inp: Image.Image, out: Image.Image) -> dict:
    a = np.asarray(inp, np.float32)
    b = np.asarray(out.resize(inp.size), np.float32)
    ga, gb = a.mean(2), b.mean(2)
    return {
        "change": float(np.abs(b - a).mean()),
        "sharpness_ratio": float(laplace(gb).var() / max(laplace(ga).var(), 1e-6)),
        "contrast_ratio": float(gb.std() / max(ga.std(), 1e-6)),
    }


def _font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _comparison(rows: list[list[tuple[Image.Image, str]]], titles: list[str], heading: str, out: Path, tile_w: int = 560):
    gap, head_h, title_h, cap_h = 8, 50, 36, 30
    tiles_h = [round(tile_w * r[0][0].height / r[0][0].width) for r in rows]
    W = len(titles) * tile_w + (len(titles) - 1) * gap
    H = head_h + title_h + sum(h + cap_h for h in tiles_h)
    c = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(c)
    d.text((4, 8), heading, font=_font(28, True), fill=(20, 33, 58))
    for j, t in enumerate(titles):
        d.text((j * (tile_w + gap) + 2, head_h + 2), t, font=_font(22, True), fill=(20, 33, 58))
    y = head_h + title_h
    for row, th in zip(rows, tiles_h):
        for j, (im, cap) in enumerate(row):
            c.paste(im.resize((tile_w, th), Image.LANCZOS), (j * (tile_w + gap), y))
            d.text((j * (tile_w + gap) + 2, y + th + 3), cap, font=_font(18), fill=(90, 103, 121))
        y += th + cap_h
    out.parent.mkdir(parents=True, exist_ok=True)
    c.save(out, quality=88, optimize=True)


def _run_segment(images: list[Path], t0: float):
    import segment_repair as sr
    use_sam3 = sr.SAM3_WEIGHTS.exists()
    for i, p in enumerate(images):
        dst = SEG / p.relative_to(SRC)
        inst = (SEG_INST / p.relative_to(SRC)).with_suffix(".jpg")
        if dst.exists() and inst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        inst.parent.mkdir(parents=True, exist_ok=True)
        a = sr.analyse(_load(p), use_sam3=use_sam3)
        sr.repair(a).save(dst)
        sr.overlay(a).save(inst, quality=88)
        print(f"[segment] {i + 1}/{len(images)}  {time.perf_counter() - t0:.0f}s  {p.relative_to(SRC)}: "
              f"{sr.summary(a).split('  |')[0]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-only", action="store_true")
    ap.add_argument("--segment", action="store_true", help="also run segment_repair.py (hybrid + SAM 3)")
    args = ap.parse_args()
    models = {"ours": MODELS["ours"]} if args.ours_only else MODELS

    images = sorted(p for p in SRC.rglob("*.png"))
    print(f"{len(images)} images in {len({p.parent for p in images})} scenarios", flush=True)

    # 1. Run the models (skips outputs that already exist, so the run can resume).
    t0 = time.perf_counter()
    for name, (weights, root) in models.items():
        for i, p in enumerate(images):
            dst = root / p.relative_to(SRC)
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            out, _ = derainnet.derain(_load(p), weights)
            out.save(dst)
            print(f"[{name}] {i + 1}/{len(images)}  {time.perf_counter() - t0:.0f}s  {p.relative_to(SRC)}", flush=True)
    if args.segment:
        _run_segment(images, t0)

    # 2. Measurements and comparison images, one per scenario folder.
    roots = {"released": RELEASED, "ours": OURS} if not args.ours_only else {"ours": OURS}
    if args.segment:
        roots["segment"] = SEG
    comp_dir = SEG_COMP if args.segment else COMP
    rows_csv, per_scenario = [], defaultdict(lambda: defaultdict(list))
    by_folder = defaultdict(list)
    for p in images:
        by_folder[p.parent].append(p)
    for folder, paths in sorted(by_folder.items()):
        rel = folder.relative_to(SRC)
        scenario = " - ".join(rel.parts)
        rows = []
        for p in paths:
            inp = _load(p)
            row = [(inp, p.stem[:40])]
            if args.segment:
                row.append((Image.open((SEG_INST / p.relative_to(SRC)).with_suffix(".jpg")).convert("RGB"),
                            "orange glare · purple streak · blue drop · grey road"))
            for name, root in roots.items():
                out = Image.open(root / p.relative_to(SRC)).convert("RGB")
                s = _stats(inp, out)
                rows_csv.append({"scenario": scenario, "image": p.name, "model": name, **{k: round(v, 3) for k, v in s.items()}})
                for k, v in s.items():
                    per_scenario[scenario][f"{name}_{k}"].append(v)
                row.append((out, f"change {s['change']:.1f} · sharp ×{s['sharpness_ratio']:.2f} · contrast ×{s['contrast_ratio']:.2f}"))
            rows.append(row)
        titles = ["Rainy input"] + (["Instances found"] if args.segment else []) + [TITLES[n] for n in roots]
        _comparison(rows, titles, scenario + f"  ({len(paths)} images)", comp_dir / (scenario.replace("/", "-") + ".jpg"),
                    tile_w=480 if args.segment else 560)
        print(f"comparison: {scenario}", flush=True)

    comp_dir.mkdir(parents=True, exist_ok=True)
    with open(comp_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    summary = {sc: {k: round(float(np.mean(v)), 3) for k, v in d.items()} | {"images": len(next(iter(d.values()))) }
               for sc, d in per_scenario.items()}
    (comp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
