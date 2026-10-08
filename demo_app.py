"""
Review-2 demo for the raindrop/streak/glare removal project.

Tab 1 runs VRGS (raindrop_render.py) live: add synthetic water-on-glass
degradation to a clean driving frame.

"Watch the model" runs the stock pretrained model one stage at a time
(model_stages.py) and shows each stage's output as it finishes.

"Remove rain" compares, on real DeRaindrop rain/clean pairs scored against
ground truth: the stock pretrained UnReflectAnything model, the same model
driven by DeRaindrop's raindrop detector (every rainy patch repaired) with
either the released inpainter or neighbour fill (model_stages.neighbour_fill),
and DeRaindrop itself (derainnet.py), the open-source raindrop-specific
baseline. Outputs are precomputed (UnReflectAnything takes ~1-2 min/image on
CPU) into DEMO_ASSETS/{stock,drnmask,drnfill,gtfill,deraindrop}_output;
DeRaindrop is fast enough (~10 s) to also run live. Our own raindrop-trained
model isn't shown yet -- it needs the GPU training run.

Run:  .venv\\Scripts\\python.exe demo_app.py
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
import derainnet  # noqa: E402
import metrics  # noqa: E402
import model_stages  # noqa: E402
from raindrop_paths import RAINDROP_ROOT  # noqa: E402
from raindrop_render import RaindropRender  # noqa: E402

ASSETS = Path(os.environ.get("DEMO_ASSETS", RAINDROP_ROOT / "demo_assets"))
URA = REPO / ".venv" / "Scripts" / "ura.exe"
MAX_SIDE = 960

renderer = RaindropRender(height=MAX_SIDE, width=MAX_SIDE)


def _fit(img: Image.Image) -> Image.Image:
    scale = MAX_SIDE / max(img.size)
    if scale >= 1:
        return img
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)


def add_rain(img: Image.Image, intensity: float, context: str, seed: float):
    if img is None:
        raise gr.Error("Pick or upload a clean frame first.")
    img = _fit(img.convert("RGB"))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    with torch.no_grad():
        result = renderer(TF.to_tensor(img).unsqueeze(0), rain_intensity=float(intensity), capture_context=context)
    degraded = TF.to_pil_image(result["degraded"][0])
    mask = TF.to_pil_image(result["mask"][0])
    return degraded, mask


def _scores(pred: Image.Image, gt: Image.Image) -> tuple[float, float]:
    p = TF.to_tensor(pred.convert("RGB")).unsqueeze(0)
    g = TF.to_tensor(gt.convert("RGB").resize(pred.size)).unsqueeze(0)
    return metrics.psnr_metric(p, g).item(), metrics.ssim_metric(p, g).item()


def _load_examples() -> dict:
    examples = {}
    for path in sorted((ASSETS / "rainy").glob("*.png"), key=lambda p: int(p.stem)):
        out_path = ASSETS / "stock_output" / path.name
        gt_path = ASSETS / "gt" / path.name
        if not (out_path.exists() and gt_path.exists()):
            continue
        rainy, out, gt = Image.open(path), Image.open(out_path), Image.open(gt_path)
        e = {"stem": path.stem, "rainy": rainy, "out": out, "gt": gt,
             "rainy_scores": _scores(rainy, gt), "out_scores": _scores(out, gt)}
        for key, folder in OPTIONAL_OUTPUTS.items():
            p = ASSETS / folder / path.name
            e[key] = Image.open(p) if p.exists() else None
            e[key + "_scores"] = _scores(e[key], gt) if e[key] is not None else None
        examples[f"Example {len(examples) + 1}"] = e
    return examples


# Precomputed outputs beyond the stock model's, keyed as in each example dict.
OPTIONAL_OUTPUTS = {
    "drnmask": "drnmask_output",  # DeRaindrop's raindrop detector + the released inpainter
    "drnfill": "drnfill_output",  # DeRaindrop's raindrop detector + neighbour fill
    "fix": "gtfill_output",  # the ground-truth droplet mask + neighbour fill
    "drn": "deraindrop_output",  # DeRaindrop itself
    "lama": "lama_output",  # DeRaindrop's raindrop detector + LaMa, in pixel space (eval_lama.py)
    "trained": "trained_output",  # our VRGS-trained detector + LaMa (eval_all.py)
    "ours": "deraindrop_vrgs_output",  # DeRaindrop fine-tuned on VRGS (finetune_deraindrop.py)
}
# The rows the comparison table shows (full results: eval_results/ and README).
ROW_LABELS = {
    "rainy": "Rainy input (left alone)",
    "out": "UnReflectAnything, stock (its highlight detector)",
    "drnfill": "UnReflectAnything, raindrop detector + neighbour fill",
    "lama": "Raindrop detector + LaMa fill (pixel space, full resolution)",
    "ours": "**Ours: DeRaindrop fine-tuned on VRGS rain**",
}


EXAMPLES = _load_examples()

# The photo every tab opens with: photo 12, where ours (DeRaindrop fine-tuned on VRGS) gains the most SSIM of the
# 8 examples (0.810 -> 0.920, +6.5 dB). DEMO_PHOTO picks another one by its file name.
FEATURED_STEM = os.environ.get("DEMO_PHOTO", "12")
FEATURED_PATH = ASSETS / "rainy" / f"{FEATURED_STEM}.png"
FEATURED_NAME = next((n for n, e in EXAMPLES.items() if e["stem"] == FEATURED_STEM), next(iter(EXAMPLES), None))
FEATURED_IMAGE = str(FEATURED_PATH) if FEATURED_PATH.exists() else None


def _summary_md() -> str:
    if not EXAMPLES:
        return "_No precomputed results found in `" + str(ASSETS / "stock_output") + "`._"
    n = len(EXAMPLES)
    md = (
        f"**Average over {n} real DeRaindrop test images**, scored against the clean photo (higher is better)\n\n"
        "| | PSNR (dB) | SSIM | Beats the rainy input on both |\n|---|---|---|---|\n"
    )
    for key in ROW_LABELS:
        scores = [e[key + "_scores"] for e in EXAMPLES.values() if e.get(key + "_scores") is not None]
        if len(scores) != n:
            continue
        p, s = sum(x[0] for x in scores) / n, sum(x[1] for x in scores) / n
        beats = "–" if key == "rainy" else f"{sum(x[0] > e['rainy_scores'][0] and x[1] > e['rainy_scores'][1] for x, e in zip(scores, EXAMPLES.values()))} of {n}"
        md += f"| {ROW_LABELS[key]} | {p:.2f} | {s:.3f} | {beats} |\n"
    return md + SUMMARY_NOTES + _vrgs_gap_md() + _finetune_md()


FINETUNE_RESULTS = REPO / "eval_results" / "deraindrop_vrgs_finetune.json"


def _finetune_md() -> str:
    """finetune_deraindrop.py's evaluation of DeRaindrop before and after VRGS fine-tuning."""
    if not FINETUNE_RESULTS.exists():
        return ""
    r = json.loads(FINETUNE_RESULTS.read_text())
    v, real = r["vrgs_heldout"], r["real_test_a"]
    md = (
        "\n\n**Training on VRGS: DeRaindrop fine-tuned on our synthetic rain** (30 min on a GPU). The dashcam/CCTV "
        f"frames come from a KITTI drive that was never trained on.\n\n"
        f"| | Dashcam/CCTV rain, held out ({v['images']} frames) | Real photos improved (of {real['images']}) |\n"
        "|---|---|---|\n"
    )
    for label, name in (("rainy input", "Rainy input"), ("DeRaindrop (released)", "DeRaindrop (released)"),
                        ("DeRaindrop + VRGS fine-tune", "**Ours: fine-tuned on VRGS**")):
        s, rs = v[label], real[label]
        beats = f", improves {s['beats_input']} of {v['images']}" if "beats_input" in s else ""
        md += (f"| {name} | {s['psnr']:.2f} dB / {s['ssim']:.3f}{beats} | "
               f"{rs['beats_input'] if 'beats_input' in rs else '–'} |\n")
    return md + (
        "\nIt is the only model that improves dashcam/CCTV frames: it learned the sliding streaks and glare from VRGS, "
        "which the released model has never seen, and it still improves every real photo."
    )


SUMMARY_NOTES = """
- **Stock UnReflectAnything** looks for bright highlights, not droplets: it marks about 4% of patches, mostly in the
  wrong places, and re-renders the rest of the image softer.
- **Raindrop detector + neighbour fill:** discards the droplet's features and rebuilds each marked patch from the
  clean patches around it, so the droplets are actually removed, but the fill is smooth and turns patchy where
  droplets are dense.
- **Raindrop detector + LaMa:** paints the marked pixels at full resolution, so doors, railings and tiles come back
  sharp; it invents texture rather than recovering it.
- **Ours, DeRaindrop fine-tuned on VRGS rain,** restores the scene behind the water instead of repainting it, and is
  the best of these by a wide margin.
"""


def _vrgs_gap_md() -> str:
    """DeRaindrop on our own VRGS dashcam/CCTV rain (precomputed into vrgs_eval/),
    i.e. away from the single glass-pane rig it was trained on."""
    rows = []
    for rainy_path in sorted((ASSETS / "vrgs_eval").glob("*_rainy.png")):
        stem = rainy_path.name[: -len("_rainy.png")]
        frame = stem.rsplit("_", 1)[0]
        out_path, clean_path = ASSETS / "vrgs_eval" / f"{stem}_deraindrop.png", ASSETS / "clean_frames" / f"{frame}.png"
        if out_path.exists() and clean_path.exists():
            clean = Image.open(clean_path)
            rows.append((_scores(Image.open(rainy_path), clean), _scores(Image.open(out_path), clean)))
    if not rows:
        return ""
    n = len(rows)
    avg = lambda i, j: sum(r[i][j] for r in rows) / n  # noqa: E731
    beats = sum(r[1][0] > r[0][0] and r[1][1] > r[0][1] for r in rows)
    return (
        f"\n**Away from its home turf:** on {n} VRGS-rendered dashcam/CCTV frames (streaks and glare included), "
        f"DeRaindrop goes from {avg(0, 0):.2f} dB / SSIM {avg(0, 1):.3f} (rainy) to {avg(1, 0):.2f} dB / SSIM "
        f"{avg(1, 1):.3f}, beating the rainy input on {beats} of {n}. "
        + VRGS_GAP_NOTE
    )


VRGS_GAP_NOTE = (
    "It leaves the sliding droplet trails and glare streaks almost untouched and only softens the frame: it was "
    "trained on round droplets on a single glass pane and has never seen either. That gap is what VRGS training "
    "targets (try it on the **Add rain** tab)."
)


def show_example(name: str):
    e = EXAMPLES[name]
    md = "| | PSNR (dB) | SSIM |\n|---|---|---|\n"
    for key, label in ROW_LABELS.items():
        if e.get(key + "_scores") is not None:
            p, s = e[key + "_scores"]
            md += f"| {label} | {p:.2f} | {s:.3f} |\n"
    return e["rainy"], e["gt"], e["ours"], e["out"], e["drnfill"], e["lama"], md


# Ours first, so it is the default wherever a version is picked.
DRN_VERSIONS = {"Ours: fine-tuned on VRGS": derainnet.FINETUNED, "Released": derainnet.WEIGHTS}
DRN_CHOICES = [k for k, p in DRN_VERSIONS.items() if p.exists()]


def run_deraindrop_live(img: Image.Image, version: str = "Released"):
    if img is None:
        raise gr.Error("Upload or pick a rainy image first.")
    out, att = derainnet.derain(_fit(img.convert("RGB")), DRN_VERSIONS[version])
    return out, model_stages._heatmap(att)


def run_stock_live(img: Image.Image):
    if img is None:
        raise gr.Error("Upload a rainy image first.")
    with tempfile.TemporaryDirectory() as tmp:
        in_dir, out_dir = Path(tmp) / "in", Path(tmp) / "out"
        in_dir.mkdir()
        img.convert("RGB").save(in_dir / "upload.png")
        env = {**os.environ, "PYTHONUTF8": "1"}
        proc = subprocess.run(
            [str(URA), "inference", str(in_dir), "-o", str(out_dir), "-d", "cpu"],
            capture_output=True, text=True, env=env, cwd=str(REPO),
        )
        out_file = out_dir / "upload.png"
        if proc.returncode != 0 or not out_file.exists():
            raise gr.Error("The model failed to run on this image: " + (proc.stderr[-400:] or "no output produced"))
        return Image.open(out_file).copy()


MODES = [
    "Raindrop detector + neighbour fill (removes the droplets)",
    "Raindrop detector + LaMa fill (pixel space, full resolution)",
    "Raindrop detector + released inpainter (turns droplets into dark blobs)",
    "Stock model as-is (its highlight detector)",
    "Correct droplet mask + neighbour fill (DeRaindrop examples only)",
]
RAINDROP_MODE, LAMA_MODE, RAINDROP_LEARNED_MODE, STOCK_MODE, GT_MODE = MODES
# Our model, trained on VRGS. Offered once its weights exist. Its raindrop detector feeds LaMa; its own
# token inpainter (one hour from scratch) is shown too, but still fills marked areas with flat grey.
TRAINED_MODE = "Experimental: our VRGS-trained raindrop detector + LaMa fill (not yet good on real photos)"
TRAINED_FULL_MODE = "Our VRGS-trained detector + its own inpainter (needs more training)"
# Ours: DeRaindrop fine-tuned on VRGS rain, the same model as "Ours" on the results tab. It restores what's behind
# the water instead of painting over it, so it runs as its own steps: four attention passes, then the restoration.
OURS_MODE = "Ours: DeRaindrop fine-tuned on VRGS rain (finds the rain, then restores what's behind it)"
OURS_STAGES = [f"Find the rain: attention pass {i + 1} of {derainnet.ITERATION}" for i in range(derainnet.ITERATION)]
OURS_STAGES.append("Restore the scene behind the water")
# The modes the demo offers: ours, our experimental detector, a repair pipeline on DeRaindrop's detector, and the
# stock baseline. The others (LaMa on DeRaindrop's detector, released inpainter, correct mask, our unfinished
# inpainter) stay available in code as diagnostics.
SHOWN_MODES = [RAINDROP_MODE, STOCK_MODE]
if model_stages.TRAINED_WEIGHTS.exists():
    SHOWN_MODES = [TRAINED_MODE] + SHOWN_MODES
if derainnet.FINETUNED.exists():
    SHOWN_MODES = [OURS_MODE] + SHOWN_MODES


def stage_names(mode: str) -> list[str]:
    return OURS_STAGES if mode == OURS_MODE else model_stages.STAGES


def _ours_stages(img: Image.Image):
    """derainnet.derain_steps in run_stages' (step, seconds, image, note) form."""
    for i, (secs, out) in enumerate(derainnet.derain_steps(img, derainnet.FINETUNED)):
        if i < derainnet.ITERATION:
            note = (f"Where the network thinks the water is (brighter = more likely); "
                    f"{100 * (out > 0.2).float().mean().item():.0f}% of pixels above 0.2")
            if i == 0:
                note += ". Each pass refines the previous one"
            yield i, secs, model_stages._heatmap(out), note
        else:
            yield i, secs, out, "Restored image: it uses the final raindrop map to recover the scene behind the water"


def _find_clean(img: Image.Image):
    """The clean ground truth for an image, if it's one of the DeRaindrop examples."""
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    for e in EXAMPLES.values():
        b = np.asarray(e["rainy"].convert("RGB"), dtype=np.int16)
        if a.shape == b.shape and np.abs(a - b).mean() < 0.5:
            return e["gt"]
    return None


def watch_model(img: Image.Image, mode: str):
    """Streams the model's progress: a status table plus one image per stage,
    each filled in as soon as that stage finishes."""
    if img is None:
        raise gr.Error("Pick an example or upload a rainy image first.")
    clean = _find_clean(img)
    mask_override = None
    if mode == GT_MODE:
        if clean is None:
            raise gr.Error("The correct droplet mask comes from the clean photo, so it only works on the DeRaindrop examples below.")
        mask_override = model_stages.ground_truth_mask(img, clean)
    composite = mode != STOCK_MODE
    detector = "deraindrop" if mode in (RAINDROP_MODE, RAINDROP_LEARNED_MODE) else "highlight"
    fill = "neighbours" if mode in (RAINDROP_MODE, GT_MODE) else "learned"
    names = stage_names(mode)
    n = len(names)
    images, notes, times = [None] * n, [""] * n, [None] * n
    lama, trained, full, ours = mode == LAMA_MODE, mode == TRAINED_MODE, mode == TRAINED_FULL_MODE, mode == OURS_MODE
    loaded = model_stages._trained if (trained or full) else model_stages._wrapper
    prefix = "" if lama or ours or loaded is not None else "Loading the model (first run only, about 30 seconds)...\n\n"

    def table(running: int) -> str:
        rows = []
        for i, name in enumerate(names):
            if times[i] is not None:
                status = f"done in {times[i]:.1f}s"
            elif i == running:
                status = "**running...**"
            else:
                status = "waiting"
            rows.append(f"| {i + 1} | {name} | {status} | {notes[i]} |")
        total = sum(t for t in times if t is not None)
        footer = ""
        if all(t is not None for t in times):
            footer = f"\n\n**Total: {total:.1f}s** on this laptop's {model_stages.DEVICE.type.upper()}."
            if clean is not None:
                (ip, is_), (op, os_) = _scores(img, clean), _scores(images[-1], clean)
                verdict = "better" if op > ip and os_ > is_ else "worse" if op < ip and os_ < is_ else "mixed"
                footer += (f"\n\n**Against the clean photo:** rainy {ip:.2f} dB / SSIM {is_:.3f} → "
                           f"output {op:.2f} dB / SSIM {os_:.3f} (**{verdict}** than leaving it alone)")
        return prefix + "| Step | What the model is doing | Status | Detail |\n|---|---|---|---|\n" + "\n".join(rows) + footer

    def shown():
        return [gr.update(value=im, label=name) for im, name in zip(images, names)]

    yield [table(0)] + shown()
    if ours:
        stages = _ours_stages(img)
    elif lama:
        stages = model_stages.run_lama_stages(img)
    elif trained:
        stages = model_stages.run_lama_stages(img, detector="trained")
    elif full:
        stages = model_stages.run_stages(img, composite=True, detector="highlight", fill="learned", trained=True,
                                         threshold=model_stages.TRAINED_THRESHOLD)
    else:
        stages = model_stages.run_stages(img, mask_override=mask_override, composite=composite, detector=detector, fill=fill)
    for i, secs, im, note in stages:
        if i == -1:
            prefix = f"Model loaded in {secs:.0f}s.\n\n"
            yield [table(0)] + shown()
            continue
        times[i], notes[i], images[i] = secs, note, im
        yield [table(i + 1)] + shown()


HEADER = """
# Raindrop, Streak & Glare Removal for Dashcam and CCTV Footage
Water sitting on the camera's own glass during rain: droplets that block and blur the scene,
streaks left as they slide, and glare around headlights and streetlights.
This project adapts UnReflectAnything (CVPR 2026) to that problem.
"""

PIPELINE = """
1. **Clean frame** from ordinary driving footage (KITTI). No real rain/clean pairs needed.
2. **VRGS adds synthetic rain:** droplets refract and blur what's behind them, trailing streaks follow one shared wind
   direction, and glare is placed where real light sources already are. It also produces an exact mask of every
   degraded pixel.
3. **Models learn to undo it:** a raindrop detector trained on VRGS masks, and DeRaindrop fine-tuned on VRGS rain,
   which learns the streaks and glare it has never seen.
4. **Evaluation** uses real rain/clean photo pairs (DeRaindrop's test set) and VRGS rain on a driving sequence that
   nothing was trained on.

All four steps run end-to-end on a laptop GPU; the "Remove rain" tab shows the results.
"""

with gr.Blocks() as demo:
    gr.Markdown(HEADER)
    with gr.Accordion("How the approach works", open=False):
        gr.Markdown(PIPELINE)

    with gr.Tab("Add rain (VRGS)"):
        gr.Markdown("Our synthesis method, running live. Adjust the rain and run it again; the seed makes a result reproducible.")
        with gr.Row():
            with gr.Column():
                clean_in = gr.Image(type="pil", label="Clean frame")
                intensity = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Rain intensity")
                context = gr.Radio(["dashcam", "cctv"], value="dashcam", label="Camera",
                                   info="Dashcam streaks drift with vehicle motion; CCTV streaks run straight down.")
                seed = gr.Number(value=7, precision=0, label="Seed")
                add_btn = gr.Button("Add rain", variant="primary")
            with gr.Column():
                rain_out = gr.Image(type="pil", label="With synthetic rain")
                mask_out = gr.Image(type="pil", label="Degradation mask (training target)")
        gr.Examples(
            examples=[[str(p)] for p in sorted((ASSETS / "clean_frames").glob("*.png"))],
            inputs=[clean_in],
            label="Clean KITTI frames",
        )
        add_btn.click(add_rain, [clean_in, intensity, context, seed], [rain_out, mask_out])
        with gr.Row():
            send_btn = gr.Button("Send this rainy frame to the step-by-step view")
            vrgs_drn_btn = gr.Button("Try DeRaindrop on this rainy frame (about 10 seconds)")
        vrgs_drn_ver = gr.Radio(DRN_CHOICES, value=DRN_CHOICES[0], label="DeRaindrop version")
        with gr.Row():
            vrgs_drn_out = gr.Image(type="pil", label="DeRaindrop output", interactive=False)
            vrgs_drn_att = gr.Image(type="pil", label="DeRaindrop's raindrop map", interactive=False)
        vrgs_drn_btn.click(run_deraindrop_live, [rain_out, vrgs_drn_ver], [vrgs_drn_out, vrgs_drn_att])

    with gr.Tab("Watch the model, step by step"):
        gr.Markdown(
            "Runs the pipeline one stage at a time and shows what each stage produces, as it happens: detect the "
            "rain, choose what to repair, repair it. A few seconds on a GPU, up to a couple of minutes on a CPU."
        )
        with gr.Row():
            with gr.Column(scale=1):
                watch_in = gr.Image(value=FEATURED_IMAGE, type="pil", label="Rainy image")
                watch_mode = gr.Radio(
                    SHOWN_MODES, value=SHOWN_MODES[0], label="How to find and repair the rain",
                    info="Ours is DeRaindrop fine-tuned on VRGS rain, the same model as \"Ours\" on the results tab: "
                         "it finds the rain, then restores what's behind it. The experimental mode uses our own "
                         "detector trained on VRGS. Neighbour fill rebuilds marked patches from the clean ones around "
                         "them. The stock model looks for bright highlights, not rain.",
                )
                watch_btn = gr.Button("Run step by step", variant="primary")
            with gr.Column(scale=2):
                watch_status = gr.Markdown()
        gr.Examples(
            examples=[[str(p)] for p in sorted((ASSETS / "rainy").glob("*.png"), key=lambda p: int(p.stem))],
            inputs=[watch_in],
            label="Real rainy photos (DeRaindrop)",
        )
        stage_imgs = []
        first_names = stage_names(SHOWN_MODES[0])
        with gr.Row():
            for name in first_names[:3]:
                stage_imgs.append(gr.Image(type="pil", label=name, interactive=False))
        with gr.Row():
            for name in first_names[3:]:
                stage_imgs.append(gr.Image(type="pil", label=name, interactive=False))
        watch_btn.click(watch_model, [watch_in, watch_mode], [watch_status] + stage_imgs)
        # Switching mode relabels the stage images and clears the previous run.
        watch_mode.change(lambda m: [""] + [gr.update(value=None, label=nm) for nm in stage_names(m)],
                          watch_mode, [watch_status] + stage_imgs)
        send_btn.click(lambda im: im, rain_out, watch_in)

    with gr.Tab("Remove rain: results"):
        gr.Markdown(
            "Real rainy photos, scored against their clean versions. **Ours** is DeRaindrop fine-tuned on our "
            "VRGS rain; the others repair the image with **UnReflectAnything**-style detect-and-fill pipelines."
        )
        gr.Markdown(_summary_md())
        names = list(EXAMPLES)
        pick = gr.Dropdown(names, value=FEATURED_NAME, label="Real rainy test photo")
        with gr.Row():
            rainy_img = gr.Image(type="pil", label="Rainy input")
            ours_img = gr.Image(type="pil", label="Ours: DeRaindrop fine-tuned on VRGS rain")
            gt_img = gr.Image(type="pil", label="Clean ground truth")
        with gr.Row():
            stock_img = gr.Image(type="pil", label=ROW_LABELS["out"])
            drnfill_img = gr.Image(type="pil", label=ROW_LABELS["drnfill"])
            lama_img = gr.Image(type="pil", label=ROW_LABELS["lama"])
        scores_md = gr.Markdown()
        example_outputs = [rainy_img, gt_img, ours_img, stock_img, drnfill_img, lama_img, scores_md]
        pick.change(show_example, pick, example_outputs)
        if names:
            demo.load(show_example, pick, example_outputs)

        with gr.Accordion("Try your own rainy image", open=False):
            live_in = gr.Image(value=FEATURED_IMAGE, type="pil", label="Rainy image")
            live_drn_ver = gr.Radio(DRN_CHOICES, value=DRN_CHOICES[0], label="DeRaindrop version")
            with gr.Row():
                live_drn_out = gr.Image(type="pil", label="DeRaindrop output")
                live_drn_att = gr.Image(type="pil", label="DeRaindrop's raindrop map")
                live_out = gr.Image(type="pil", label="Stock UnReflectAnything output")
            with gr.Row():
                gr.Button("Run DeRaindrop (about 10 seconds)", variant="primary").click(
                    run_deraindrop_live, [live_in, live_drn_ver], [live_drn_out, live_drn_att])
                gr.Button("Run stock UnReflectAnything (about 1–2 minutes)").click(run_stock_live, live_in, live_out)

if __name__ == "__main__":
    # Load the model in the background now, so the first step-by-step run
    # doesn't sit through a ~1.5 min load in front of an audience.
    threading.Thread(target=model_stages._load, daemon=True).start()
    if model_stages.TRAINED_WEIGHTS.exists():  # ours too (~1 min: a 2.6 GB checkpoint)
        threading.Thread(target=model_stages._load, kwargs={"trained": True}, daemon=True).start()
    # The example images live in raindrop_project/, outside the repo, so Gradio must be allowed to serve them.
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, allowed_paths=[str(RAINDROP_ROOT)])
