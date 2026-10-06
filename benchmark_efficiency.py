"""
Measures the numbers for the review's "Computational Efficiency & Resource
Utilization" slide on this machine: parameters, FLOPs, latency per stage,
throughput and memory for

  * stock UnReflectAnything (highlight detector + released inpainter),
  * this project's current pipeline (DeRaindrop's raindrop detector +
    UnReflectAnything's encoder/decoder with neighbour fill),
  * DeRaindrop on its own,
  * the VRGS renderer (training-data synthesis).

FLOPs come from torch's FlopCounterMode (matmuls, convolutions and attention;
elementwise ops are ignored, as is usual). Latency is the median of --runs
timed runs after one warm-up, batch size 1, so throughput = 1 / latency.
Memory is the process's resident set: the increase from loading the weights,
and the peak above that while each stage runs (sampled every 5 ms).

    python benchmark_efficiency.py --runs 3
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import threading
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode
from torchvision.transforms import functional as TF

import derainnet
import model_stages as ms
from raindrop_render import RaindropRender

ASSETS = Path(r"C:\Users\sachd\Downloads\raindrop_project\demo_assets")
RESULTS = Path(__file__).parent / "eval_results" / "efficiency.json"
PROC = psutil.Process()


def _rss_mb() -> float:
    return PROC.memory_info().rss / 2**20


class PeakRSS:
    """Peak resident memory above the level at entry, sampled in a thread."""

    def __enter__(self):
        self.base = self.peak = _rss_mb()
        self._stop = threading.Event()

        def poll():
            while not self._stop.is_set():
                self.peak = max(self.peak, _rss_mb())
                time.sleep(0.005)

        self._t = threading.Thread(target=poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, _rss_mb())
        self.extra = self.peak - self.base


def measure(fn, runs: int) -> dict:
    """FLOPs (one counted pass), median latency over `runs`, peak extra memory."""
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            fn()
        flops = fc.get_total_flops()
        with PeakRSS() as mem:
            fn()  # warm-up, also gives the memory peak
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
    lat = statistics.median(times)
    return {"gflops": round(flops / 1e9, 1), "latency_s": round(lat, 2),
            "throughput_img_per_s": round(1 / lat, 3), "peak_extra_mb": round(mem.extra)}


def n_params(m: torch.nn.Module) -> float:
    return round(sum(p.numel() for p in m.parameters()) / 1e6, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    img = Image.open(ASSETS / "rainy" / "12.png").convert("RGB")  # 720×480, a real DeRaindrop test photo
    out = {
        "machine": {
            "cpu": platform.processor(),
            "threads": torch.get_num_threads(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "input": f"{img.width}x{img.height} photo; UnReflectAnything resizes it to its 896x896 working size",
    }

    rss0 = _rss_mb()
    ms._load()
    drn = derainnet.load()
    out["weights_rss_mb"] = {"unreflectanything_and_deraindrop": round(_rss_mb() - rss0)}

    inner, side, patch = ms._wrapper._model, ms._wrapper.image_size, ms._wrapper._model.patch_size
    decoder = next(d for n, d in inner.decoders.items() if n != "highlight")
    out["params_m"] = {
        "dinov3_encoder (frozen)": n_params(inner.dinov3),
        "highlight_detector_decoder": n_params(inner.decoders["highlight"]),
        "token_inpainter": n_params(inner.token_inpaint),
        "image_decoder": n_params(decoder),
        "unreflectanything_total": n_params(inner),
        "deraindrop_generator": n_params(drn),
        "deraindrop_attention_branch": round(sum(
            p.numel() for n, p in drn.named_parameters() if n.startswith(("det_conv", "conv_i", "conv_f", "conv_g", "conv_o"))
        ) / 1e6, 2),
    }

    # Shared intermediate results so each stage is timed on its own.
    x = TF.resize(TF.to_tensor(img).unsqueeze(0), [side, side], antialias=True)
    with torch.no_grad():
        tokens = inner.dinov3(inner.dinov3.preprocess_image(x))["selected_hidden_states"]
        grid = int(tokens[0].shape[1] ** 0.5)
        att = F.interpolate(derainnet.attention_map(img), size=(side, side), mode="bilinear", align_corners=False)
        pix = F.max_pool2d((att > 0.2).float(), 9, 1, 4) > 0.5
        hole = ms._M.pixel_mask_to_patch_mask(pix, patch_size=patch, threshold=0, invert=False, soft=False)
        completed = [torch.where(hole.unsqueeze(-1), ms.neighbour_fill(T, hole, grid), T) for T in tokens]
    print(f"{100 * hole.float().mean():.0f}% of patches marked by the raindrop detector", flush=True)

    stages = {
        "dinov3_encode": lambda: inner.dinov3(inner.dinov3.preprocess_image(x)),
        "highlight_detector": lambda: inner.decoders["highlight"](tokens),
        "raindrop_detector (DeRaindrop attention)": lambda: derainnet.attention_map(img),
        "released_inpainter (4 layers)": lambda: [inner.token_inpaint(T, ~hole) for T in tokens],
        "neighbour_fill (4 layers)": lambda: [ms.neighbour_fill(T, hole, grid) for T in tokens],
        "image_decoder": lambda: decoder(completed),
        "deraindrop_full": lambda: derainnet.derain(img),
    }
    out["stages"] = {}
    for name, fn in stages.items():
        out["stages"][name] = measure(fn, args.runs)
        print(name, out["stages"][name], flush=True)

    s = out["stages"]
    def pipeline(parts):
        return {"gflops": round(sum(s[p]["gflops"] for p in parts), 1),
                "latency_s": round(sum(s[p]["latency_s"] for p in parts), 2),
                "throughput_img_per_s": round(1 / sum(s[p]["latency_s"] for p in parts), 3),
                "peak_extra_mb": max(s[p]["peak_extra_mb"] for p in parts),
                "stages": parts}
    out["pipelines"] = {
        "stock_unreflectanything": pipeline(["dinov3_encode", "highlight_detector", "released_inpainter (4 layers)", "image_decoder"]),
        "ours_raindrop_detector_neighbour_fill": pipeline(["raindrop_detector (DeRaindrop attention)", "dinov3_encode",
                                                           "neighbour_fill (4 layers)", "image_decoder"]),
        "deraindrop": pipeline(["deraindrop_full"]),
    }

    # VRGS: synthesis cost per training frame (960-wide KITTI frame, mid intensity).
    clean = Image.open(sorted((ASSETS / "clean_frames").glob("*.png"))[0]).convert("RGB")
    clean = clean.resize((960, round(clean.height * 960 / clean.width)), Image.BICUBIC)
    renderer = RaindropRender(height=960, width=960)
    frame = TF.to_tensor(clean).unsqueeze(0)
    vr = measure(lambda: renderer(frame, rain_intensity=0.6, capture_context="dashcam"), args.runs)
    vr["frame"] = f"{clean.width}x{clean.height}"
    out["vrgs_render"] = vr
    print("vrgs_render", vr, flush=True)

    out["process_peak_rss_mb"] = round(PROC.memory_info().peak_wset / 2**20) if hasattr(PROC.memory_info(), "peak_wset") else None
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["pipelines"], indent=2))
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
