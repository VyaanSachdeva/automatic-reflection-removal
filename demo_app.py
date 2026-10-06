"""
Review-2 demo for the raindrop/streak/glare removal project.

Tab 1 runs VRGS (raindrop_render.py) live: add synthetic water-on-glass
degradation to a clean driving frame.

"Watch the model" runs the stock pretrained model one stage at a time
(model_stages.py) and shows each stage's output as it finishes.

"Remove rain" shows the stock pretrained UnReflectAnything model on real DeRaindrop
rain/clean pairs, scored against ground truth. Outputs are precomputed
(CPU inference is ~1-2 min/image) by running `ura inference` over
DEMO_ASSETS/rainy into DEMO_ASSETS/stock_output; live upload is available
but slow. Our own raindrop-trained model isn't shown yet -- it needs the
GPU training run.

Run:  .venv\\Scripts\\python.exe demo_app.py
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms import functional as TF

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
import metrics  # noqa: E402
import model_stages  # noqa: E402
from raindrop_render import RaindropRender  # noqa: E402

ASSETS = Path(os.environ.get("DEMO_ASSETS", r"C:\Users\sachd\Downloads\raindrop_project\demo_assets"))
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
        examples[f"Example {len(examples) + 1}"] = {
            "rainy": rainy,
            "out": out,
            "gt": gt,
            "rainy_scores": _scores(rainy, gt),
            "out_scores": _scores(out, gt),
        }
    return examples


EXAMPLES = _load_examples()


def _summary_md() -> str:
    if not EXAMPLES:
        return "_No precomputed results found in `" + str(ASSETS / "stock_output") + "`._"
    n = len(EXAMPLES)
    rp = sum(e["rainy_scores"][0] for e in EXAMPLES.values()) / n
    rs = sum(e["rainy_scores"][1] for e in EXAMPLES.values()) / n
    op = sum(e["out_scores"][0] for e in EXAMPLES.values()) / n
    os_ = sum(e["out_scores"][1] for e in EXAMPLES.values()) / n
    worse = sum(
        e["out_scores"][0] < e["rainy_scores"][0] and e["out_scores"][1] < e["rainy_scores"][1]
        for e in EXAMPLES.values()
    )
    return (
        f"**Average over {n} real DeRaindrop test images** (higher is better)\n\n"
        "| | PSNR (dB) | SSIM |\n|---|---|---|\n"
        f"| Rainy input vs. clean ground truth | {rp:.2f} | {rs:.3f} |\n"
        f"| Stock UnReflectAnything output vs. clean ground truth | {op:.2f} | {os_:.3f} |\n\n"
        f"**On {worse} of {n} images the stock model scores lower than the untouched rainy input on both metrics.** "
        "It leaves the droplets in place and softens the rest of the image, so a model trained for this specific "
        "degradation is needed."
    )


def show_example(name: str):
    e = EXAMPLES[name]
    (rp, rs), (op, os_) = e["rainy_scores"], e["out_scores"]
    md = (
        "| | PSNR (dB) | SSIM |\n|---|---|---|\n"
        f"| Rainy input | {rp:.2f} | {rs:.3f} |\n"
        f"| Stock model output | {op:.2f} | {os_:.3f} |\n"
    )
    return e["rainy"], e["out"], e["gt"], md


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


def watch_model(img: Image.Image):
    """Streams the model's progress: a status table plus one image per stage,
    each filled in as soon as that stage finishes."""
    if img is None:
        raise gr.Error("Pick an example or upload a rainy image first.")
    n = len(model_stages.STAGES)
    images, notes, times = [None] * n, [""] * n, [None] * n
    prefix = "" if model_stages._wrapper is not None else "Loading the pretrained model (first run only, about 30 seconds)...\n\n"

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
        footer = f"\n\n**Total: {total:.1f}s** on this laptop's CPU." if all(t is not None for t in times) else ""
        return prefix + "| Step | What the model is doing | Status | Detail |\n|---|---|---|---|\n" + "\n".join(rows) + footer

    yield [table(0)] + images
    for i, secs, im, note in model_stages.run_stages(img):
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
so the "Remove rain" tab shows the stock pretrained model as the baseline to beat.
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
        send_btn = gr.Button("Send this rainy frame to the step-by-step view")

    with gr.Tab("Watch the model, step by step"):
        gr.Markdown(
            "Runs the pretrained UnReflectAnything model one stage at a time and shows what each stage produces, "
            "as it happens. A full run takes about 1–2 minutes on this laptop's CPU; the table shows where it is."
        )
        with gr.Row():
            with gr.Column(scale=1):
                watch_in = gr.Image(type="pil", label="Rainy image")
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
        watch_btn.click(watch_model, watch_in, [watch_status] + stage_imgs)
        send_btn.click(lambda im: im, rain_out, watch_in)

    with gr.Tab("Remove rain (baseline)"):
        gr.Markdown(
            "The **stock pretrained UnReflectAnything model**, built for highlights on object surfaces, run on real "
            "rainy photos. It shows how far a general-purpose remover gets on raindrops, which is the gap this project targets."
        )
        gr.Markdown(_summary_md())
        names = list(EXAMPLES)
        pick = gr.Dropdown(names, value=names[0] if names else None, label="Real DeRaindrop test image")
        with gr.Row():
            rainy_img = gr.Image(type="pil", label="Rainy input")
            stock_img = gr.Image(type="pil", label="Stock model output")
            gt_img = gr.Image(type="pil", label="Clean ground truth")
        scores_md = gr.Markdown()
        pick.change(show_example, pick, [rainy_img, stock_img, gt_img, scores_md])
        if names:
            demo.load(show_example, pick, [rainy_img, stock_img, gt_img, scores_md])

        with gr.Accordion("Try your own rainy image (slow: about 1–2 minutes on this laptop's CPU)", open=False):
            with gr.Row():
                live_in = gr.Image(type="pil", label="Rainy image")
                live_out = gr.Image(type="pil", label="Stock model output")
            gr.Button("Run the stock model").click(run_stock_live, live_in, live_out)

if __name__ == "__main__":
    # Load the model in the background now, so the first step-by-step run
    # doesn't sit through a ~1.5 min load in front of an audience.
    threading.Thread(target=model_stages._load, daemon=True).start()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
