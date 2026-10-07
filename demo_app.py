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
        e = {"rainy": rainy, "out": out, "gt": gt,
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
    "trained": "trained_output",  # our VRGS-trained detector + inpainter (eval_all.py)
}
ROW_LABELS = {
    "rainy": "Rainy input (left alone)",
    "out": "UnReflectAnything, stock (its highlight detector)",
    "drnmask": "UnReflectAnything, raindrop detector + released inpainter",
    "drnfill": "UnReflectAnything, raindrop detector + neighbour fill",
    "lama": "Raindrop detector + LaMa fill (pixel space, full resolution)",
    "trained": "Ours: VRGS-trained raindrop detector + LaMa fill",
    "fix": "UnReflectAnything, correct droplet mask + neighbour fill (upper bound on detection)",
    "drn": "DeRaindrop (CVPR 2018, open-source raindrop model)",
}


EXAMPLES = _load_examples()


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
    md = (
        "\n\n**First step on the baseline: DeRaindrop fine-tuned on VRGS rain** "
        f"({r.get('run', 'short CPU run')}). Held-out frames were never trained on.\n\n"
        "| | VRGS dashcam/CCTV, held out | Real photos (DeRaindrop test set) |\n|---|---|---|\n"
    )
    for label in ("rainy input", "DeRaindrop (released)", "DeRaindrop + VRGS fine-tune"):
        cells = []
        for split in ("vrgs_heldout", "real_test_a"):
            s = r[split][label]
            beats = f" (beats input on {s['beats_input']} of {r[split]['images']})" if "beats_input" in s else ""
            cells.append(f"{s['psnr']:.2f} dB / {s['ssim']:.3f}{beats}")
        md += f"| {label[0].upper() + label[1:]} | {cells[0]} | {cells[1]} |\n"
    return md + (
        "\nAfter 30 minutes on the GPU it is the first version that helps on dashcam/CCTV rain (it removes some "
        "streaks and glare), at the cost of 0.8 dB PSNR on the real photos; SSIM there is unchanged."
    )


SUMMARY_NOTES = """
- **Stock UnReflectAnything** looks for bright highlights, not droplets: it marks about 4% of patches, mostly in the
  wrong places, and re-renders the rest of the image softer.
- **Raindrop detector + released inpainter:** DeRaindrop's raindrop map marks every rainy patch, but each droplet comes
  back as a dark blob. The released inpainter was trained with an inverted mask: at each hole it is given that hole's
  own features and learned to correct a highlight in place (mostly by darkening it), so it keeps the droplet instead
  of replacing it.
- **Raindrop detector + neighbour fill:** discards the droplet's features and rebuilds each marked patch from the
  clean patches around it, so the droplets are actually removed. The fill is smooth (it can't invent fine detail), and
  where droplets are dense the repaired areas turn soft and patchy, which still costs it against the untouched photo
  on these metrics. A trained inpainter is what closes that gap.
- **DeRaindrop**, a model trained specifically on raindrops, is the strongest baseline here. But these photos come
  from its own test set (same glass pane and camera it was trained with), so this is its best case.
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
    return e["rainy"], e["gt"], e["drn"], e["out"], e["drnmask"], e["drnfill"], e["lama"], e["trained"], md


DRN_VERSIONS = {"Released": derainnet.WEIGHTS, "Fine-tuned on VRGS": derainnet.FINETUNED}
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
TRAINED_MODE = "Our VRGS-trained raindrop detector + LaMa fill"
TRAINED_FULL_MODE = "Our VRGS-trained detector + its own inpainter (needs more training)"
if model_stages.TRAINED_WEIGHTS.exists():
    MODES = [TRAINED_MODE] + MODES + [TRAINED_FULL_MODE]


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
    n = len(model_stages.STAGES)
    images, notes, times = [None] * n, [""] * n, [None] * n
    lama, trained, full = mode == LAMA_MODE, mode == TRAINED_MODE, mode == TRAINED_FULL_MODE
    loaded = model_stages._trained if (trained or full) else model_stages._wrapper
    prefix = "" if lama or loaded is not None else "Loading the model (first run only, about 30 seconds)...\n\n"

    def table(running: int) -> str:
        rows = []
        for i, name in enumerate(model_stages.STAGES):
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
                verdict = ("**better** than leaving it alone" if op > ip and os_ > is_ else
                           "**worse** than leaving it alone" if op < ip and os_ < is_ else
                           "**mixed**: better than leaving it alone on one score, worse on the other")
                footer += (
                    f"\n\n**Against the clean photo:** rainy input {ip:.2f} dB / SSIM {is_:.3f} → "
                    f"output {op:.2f} dB / SSIM {os_:.3f} ({verdict})"
                )
        return prefix + "| Step | What the model is doing | Status | Detail |\n|---|---|---|---|\n" + "\n".join(rows) + footer

    yield [table(0)] + images
    if lama:
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
            yield [table(0)] + images
            continue
        times[i], notes[i], images[i] = secs, note, im
        yield [table(i + 1)] + images


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
3. **The model learns to undo it:** detect the degraded regions, fill them in using DINOv3 features, decode back to an image.
4. **Evaluation** uses DeRaindrop's real rain/clean photo pairs, which the model never trains on.

Steps 1–3 are built and verified end-to-end. The full-scale training run is waiting on GPU time,
so the "Remove rain" tab compares the existing models this project builds on and has to beat: stock
UnReflectAnything, and DeRaindrop, an open-source raindrop-specific model.
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
            "Runs the pretrained UnReflectAnything model one stage at a time and shows what each stage produces, "
            "as it happens. A full run takes about 1–2 minutes on this laptop's CPU; the table shows where it is."
        )
        with gr.Row():
            with gr.Column(scale=1):
                watch_in = gr.Image(type="pil", label="Rainy image")
                watch_mode = gr.Radio(
                    MODES, value=RAINDROP_MODE, label="Which regions to repair",
                    info="The raindrop detector (borrowed from DeRaindrop) marks every rainy patch; the stock one looks "
                         "for bright highlights instead. The released inpainter was trained to correct highlights in "
                         "place, so it keeps each droplet and darkens it; neighbour fill rebuilds those patches from "
                         "the clean ones around them. LaMa skips the feature space and paints the marked pixels "
                         "directly at full resolution. The correct mask shows what perfect detection would give.",
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
        with gr.Row():
            for name in model_stages.STAGES[:3]:
                stage_imgs.append(gr.Image(type="pil", label=name, interactive=False))
        with gr.Row():
            for name in model_stages.STAGES[3:]:
                stage_imgs.append(gr.Image(type="pil", label=name, interactive=False))
        watch_btn.click(watch_model, [watch_in, watch_mode], [watch_status] + stage_imgs)
        send_btn.click(lambda im: im, rain_out, watch_in)

    with gr.Tab("Remove rain: baselines compared"):
        gr.Markdown(
            "Two existing models on real rainy photos: **UnReflectAnything** (built for highlights on object surfaces; "
            "this project's architecture) and **DeRaindrop** (CVPR 2018, an open-source model trained specifically on "
            "raindrops). These are the baselines the project's VRGS-trained model has to beat."
        )
        gr.Markdown(_summary_md())
        names = list(EXAMPLES)
        pick = gr.Dropdown(names, value=names[0] if names else None, label="Real DeRaindrop test image")
        with gr.Row():
            rainy_img = gr.Image(type="pil", label="Rainy input")
            gt_img = gr.Image(type="pil", label="Clean ground truth")
            drn_img = gr.Image(type="pil", label=ROW_LABELS["drn"])
        with gr.Row():
            stock_img = gr.Image(type="pil", label=ROW_LABELS["out"])
            drnmask_img = gr.Image(type="pil", label=ROW_LABELS["drnmask"])
            drnfill_img = gr.Image(type="pil", label=ROW_LABELS["drnfill"])
            lama_img = gr.Image(type="pil", label=ROW_LABELS["lama"])
            trained_img = gr.Image(type="pil", label=ROW_LABELS["trained"])
        scores_md = gr.Markdown()
        example_outputs = [rainy_img, gt_img, drn_img, stock_img, drnmask_img, drnfill_img, lama_img, trained_img,
                           scores_md]
        pick.change(show_example, pick, example_outputs)
        if names:
            demo.load(show_example, pick, example_outputs)

        with gr.Accordion("Try your own rainy image", open=False):
            live_in = gr.Image(type="pil", label="Rainy image")
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
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
