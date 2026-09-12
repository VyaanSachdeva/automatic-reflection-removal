"""
Phase 3 baseline reproduction: paired-GT metrics (MSEm, PSNR, SSIM masked to
the highlight region, per the paper) plus Luminance Suppression Ratio (LSR),
computed on the locally-available paired datasets (SHIQ, PSD, SSHR).

Only SHIQ ships a real per-pixel highlight mask (the "_T" file); PSD and SSHR
have no ground-truth mask, so their masked metrics use a brightness-threshold
mask (utilities.inference.compute_highlight_mask) as an approximation --
labelled "approx-masked" below to distinguish from SHIQ's real-masked numbers.

NOT covered here (needs datasets not present locally -- see Phase 2 notes):
  - NIQE on non-referenced data (CroMo/Cholec80/SCARED/StereoMIS/etc.)
  - The downstream pose-estimation eval (DISK + LightGlue + MAGSAC++)

CPU-only inference is slow (~60s/image measured on this machine for the
DINOv3-Large backbone at 896x896), so this evaluates a small subset per
dataset by default. Pass --num-samples to scale up (needs a GPU to be
practical at full test-set scale).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).parent))
from utilities.inference import compute_highlight_mask  # noqa: E402
import metrics  # noqa: E402

REPO_ROOT = Path(__file__).parent
VENV_URA = REPO_ROOT / ".venv" / "Scripts" / "ura.exe"


def load_img(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        return TF.to_tensor(img.convert("RGB")).unsqueeze(0)  # [1,3,H,W]


def load_mask(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        return TF.to_tensor(img.convert("L")).unsqueeze(0)  # [1,1,H,W]


def run_inference(input_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    subprocess.run(
        [str(VENV_URA), "inference", str(input_dir), "-o", str(output_dir), "-d", "cpu"],
        check=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


def stage_inputs(pairs: list[tuple[Path, Path]], staging_dir: Path) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    for input_path, _ in pairs:
        target = staging_dir / input_path.name
        try:
            target.hardlink_to(input_path)
        except OSError:
            shutil.copy2(input_path, target)


def eval_dataset(
    name: str,
    pairs: list[tuple[Path, Path]],
    mask_paths: dict[str, Path] | None,
    work_dir: Path,
    real_mask: bool,
) -> dict:
    staging_in = work_dir / f"{name}_in"
    staging_out = work_dir / f"{name}_out"
    stage_inputs(pairs, staging_in)
    print(f"[{name}] running inference on {len(pairs)} images (CPU, this is slow)...")
    run_inference(staging_in, staging_out)

    metric_totals = {"mse": [], "psnr": [], "ssim": [], "lsr": []}
    for input_path, gt_path in pairs:
        out_path = staging_out / input_path.name
        if not out_path.exists():
            print(f"  WARNING: no output produced for {input_path.name}, skipping")
            continue
        pred = load_img(out_path)
        gt = load_img(gt_path)
        inp = load_img(input_path)
        if pred.shape != gt.shape:
            gt = torch.nn.functional.interpolate(gt, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        if pred.shape != inp.shape:
            inp = torch.nn.functional.interpolate(inp, size=pred.shape[-2:], mode="bilinear", align_corners=False)

        if mask_paths is not None and input_path.name in mask_paths:
            mask = load_mask(mask_paths[input_path.name])
            if mask.shape[-2:] != pred.shape[-2:]:
                mask = torch.nn.functional.interpolate(mask, size=pred.shape[-2:], mode="nearest")
        else:
            mask = compute_highlight_mask(inp, threshold=0.7)

        metric_totals["mse"].append(metrics.mse_metric(pred, gt, mask=mask).item())
        metric_totals["psnr"].append(metrics.psnr_metric(pred, gt, mask=mask).item())
        metric_totals["ssim"].append(metrics.ssim_metric(pred, gt, mask=mask).item())
        metric_totals["lsr"].append(metrics.luminance_suppression_ratio(inp, pred, mask=mask).item())

    n = len(metric_totals["mse"])
    result = {
        "dataset": name,
        "mask_type": "real (SHIQ _T)" if real_mask else "approx (brightness threshold)",
        "n_samples": n,
        "MSEm": sum(metric_totals["mse"]) / n if n else float("nan"),
        "PSNR": sum(metric_totals["psnr"]) / n if n else float("nan"),
        "SSIM": sum(metric_totals["ssim"]) / n if n else float("nan"),
        "LSR": sum(metric_totals["lsr"]) / n if n else float("nan"),
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=r"C:\Users\sachd\Downloads\datasets\organized")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--out", default="eval_results/phase3_baseline.json")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    work_dir = REPO_ROOT / "tmp" / "phase3_eval"
    results = []

    # --- SHIQ: real GT diffuse + real GT mask ---
    shiq_test = dataset_dir / "SHIQ" / "test"
    input_dir, gt_dir, mask_dir = shiq_test / "diffuse", shiq_test / "gt_clean", shiq_test / "highlight"
    names = sorted(p.name for p in input_dir.glob("*.png"))[: args.num_samples]
    pairs = [(input_dir / n, gt_dir / n) for n in names if (gt_dir / n).exists()]
    mask_paths = {n: mask_dir / n for n in names if (mask_dir / n).exists()}
    if pairs:
        results.append(eval_dataset("SHIQ", pairs, mask_paths, work_dir, real_mask=True))

    # --- PSD: no GT mask, approx via brightness threshold ---
    psd_test = dataset_dir / "PSD_Dataset" / "PSD_Test"
    input_dir, gt_dir = psd_test / "specular", psd_test / "diffuse"
    if input_dir.exists():
        names = sorted(p.name for p in input_dir.glob("*.png"))[: args.num_samples]
        pairs = [(input_dir / n, gt_dir / n) for n in names if (gt_dir / n).exists()]
        if pairs:
            results.append(eval_dataset("PSD", pairs, None, work_dir, real_mask=False))

    # --- SSHR: no GT mask, approx via brightness threshold ---
    sshr_root = dataset_dir / "SSHR"
    if sshr_root.exists():
        pairs = []
        for scene_dir in sorted(sshr_root.glob("test_*")):
            input_dir, gt_dir = scene_dir / "rgb", scene_dir / "diffuse"
            if not input_dir.exists():
                continue
            for p in sorted(input_dir.glob("*.jpg")):
                if (gt_dir / p.name).exists():
                    pairs.append((p, gt_dir / p.name))
                if len(pairs) >= args.num_samples:
                    break
            if len(pairs) >= args.num_samples:
                break
        if pairs:
            results.append(eval_dataset("SSHR", pairs, None, work_dir, real_mask=False))

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== Phase 3 baseline results (subset, CPU) ===")
    print(f"{'Dataset':<8} {'Mask':<28} {'N':<4} {'MSEm':<10} {'PSNR':<10} {'SSIM':<10} {'LSR':<10}")
    for r in results:
        print(
            f"{r['dataset']:<8} {r['mask_type']:<28} {r['n_samples']:<4} "
            f"{r['MSEm']:<10.5f} {r['PSNR']:<10.3f} {r['SSIM']:<10.4f} {r['LSR']:<10.4f}"
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
