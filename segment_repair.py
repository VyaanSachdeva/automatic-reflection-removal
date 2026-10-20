"""
Training-free "segment, then repair" for rain-on-glass footage: every defect in a
frame is found as its own instance, then repaired by the pretrained model that
suits its type. No model here is trained or fine-tuned for this pipeline.

Instances (analyse()):
  glare   Flare7K++ (Dai et al., NeurIPS 2022 / 2023) splits a frame into a
          flare-free scene and a flare layer. Each connected region of the flare
          layer is one glare instance: headlight bloom, lamp halos, lens/dome
          reflections. Learned from flare data, so it ignores plain bright areas
          (white walls, sky) that a brightness threshold would catch.
  drop    our DeRaindrop fine-tuned on VRGS (derainnet.py): each connected region
          of its final raindrop attention map, outside the glare, is one drop.
  streak  a raindrop-map region that is large and clearly brighter than what
          surrounds it: light smeared across the camera dome by water, or an
          out-of-focus web or strand lit by the IR lamp. Flare7K++ doesn't see
          these as flare. A large region that is *not* brighter than its
          surroundings stays a drop: either a dense cluster of real drops or
          scene content such as a puddle, which DeRaindrop's restore leaves
          largely as it is. Bright wet-road reflections at night can still be
          called streaks; telling those apart needs semantics (SAM 3, below).
  object  YOLO11-seg (COCO weights): people and vehicles, kept from being
          repainted so evidence such as plates and faces is never invented.

Optional, use_sam3=True: SAM 3 (Meta, 2025) segments concepts named in text.
  road    "wet road": real scene. No drop or streak is found inside it, so
          reflections on the road (headlights, lamps, puddles) are kept.
  streak  "glare" / "lens flare" that reach beyond the road are on the lens and
          never protected as road. Where Flare7K++ missed them (lit web
          strands, smeared lamp light) they are dimmed like streaks, since
          Flare7K++'s glare-free frame barely changes there. Ones lying almost
          entirely on the road are road reflections and are kept.
  Tested on 20 CCTV frames: "wet road" was found on all of them (0.6-0.99
  confidence); "raindrop" found a third as many drops as our DeRaindrop, and
  "spider web" was unreliable, so neither is used.

Repairs (repair()), by default a hybrid: our DeRaindrop restores the whole
frame (it also fixes faint drops too weak to be found as instances), then glare
-> the glare Flare7K++ removes is taken off, streaks -> dimmed further (not on
people or vehicles), wet road -> back to the original frame. Each is blended in
with a soft edge. repair(whole_frame=False) repairs only the instances and
leaves the rest of the frame exactly as it was.

Weights live in RAINDROP_ROOT/pretrained (see SOURCES.txt there):
  Flare7K/                    the Flare7K code, only its Uformer network is used
  flare7kpp/net_g_last.pth    Flare7K++ checkpoint
  yolo/yolo11s-seg.pt         YOLO11s-seg
  sam3/sam3.pt                SAM 3 (gated: accept Meta's licence on huggingface.co/facebook/sam3)

    python segment_repair.py <image or folder> [--out DIR] [--half] [--sam3]   # instance overlays + repaired frames
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import derainnet
from raindrop_paths import RAINDROP_ROOT

PRETRAINED = RAINDROP_ROOT / "pretrained"
FLARE_CODE = PRETRAINED / "Flare7K"
FLARE_WEIGHTS = PRETRAINED / "flare7kpp" / "net_g_last.pth"
YOLO_WEIGHTS = PRETRAINED / "yolo" / "yolo11s-seg.pt"
SAM3_WEIGHTS = PRETRAINED / "sam3" / "sam3.pt"

# COCO classes kept from being repainted
PROTECTED = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
COLOURS = {"glare": (255, 170, 30), "streak": (200, 90, 255), "drop": (40, 170, 255), "object": (60, 220, 90),
           "road": (150, 160, 175)}
# SAM 3 text prompts and the confidence each must reach
SAM3_PROMPTS = {"wet road": 0.5, "glare": 0.6, "lens flare": 0.6}


@dataclass
class Instance:
    id: int
    kind: str           # "glare", "streak", "drop", "object" or "road"
    label: str          # e.g. "glare #2", "car #1"
    mask: np.ndarray    # [H, W] bool at the frame's size
    score: float        # mean flare strength / attention / detector confidence
    repair: str = ""    # filled in by the router (next step)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.mask)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@dataclass
class Analysis:
    image: Image.Image
    instances: list[Instance] = field(default_factory=list)
    flare_free: Image.Image | None = None   # Flare7K++ scene, at the frame's size
    flare_layer: np.ndarray | None = None   # [H, W] flare strength in [0, 1]
    derained: Image.Image | None = None     # our DeRaindrop's restored frame
    rain_map: np.ndarray | None = None      # [H, W] raindrop attention in [0, 1]
    seconds: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- Flare7K++

_flare_model = None


def _load_flare():
    """Flare7K's Uformer, imported on its own: its BasicSR training framework is
    stubbed out, so only the network file runs. Weights load in safe
    (weights_only) mode."""
    global _flare_model
    if _flare_model is None:
        reg = types.ModuleType("basicsr.utils.registry")
        reg.ARCH_REGISTRY = types.SimpleNamespace(register=lambda *a, **k: (lambda c: c))
        saved = {n: sys.modules.get(n) for n in ("basicsr", "basicsr.utils", "basicsr.utils.registry")}
        sys.modules.update({"basicsr": types.ModuleType("basicsr"), "basicsr.utils": types.ModuleType("basicsr.utils"),
                            "basicsr.utils.registry": reg})
        try:
            spec = importlib.util.spec_from_file_location("flare7k_uformer", FLARE_CODE / "basicsr/archs/uformer_arch.py")
            arch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(arch)
        finally:
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m
        sd = torch.load(FLARE_WEIGHTS, map_location="cpu", weights_only=True)
        sd = sd.get("params_ema", sd.get("params", sd))
        model = arch.Uformer(img_size=512, img_ch=3, output_ch=6)
        model.load_state_dict(sd, strict=True)
        _flare_model = model.eval().to(derainnet.DEVICE)
    return _flare_model


@torch.no_grad()
def flare7kpp(img: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Returns (flare-free frame, flare layer [H, W] in [0, 1]) at the frame's size.
    As in Flare7K's test_large.py, the frame is scaled so its shorter side is
    512 and run as 512x512 tiles every 256 px, blended where they overlap."""
    model = _load_flare()
    rgb = img.convert("RGB")
    s = 512 / min(rgb.size)
    small = rgb.resize((round(rgb.width * s), round(rgb.height * s)), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(small, np.float32) / 255).permute(2, 0, 1)[None].to(derainnet.DEVICE)
    portrait = x.shape[-2] > x.shape[-1]
    if portrait:
        x = x.transpose(-1, -2)
    L = x.shape[-1]
    ends = sorted({min(e, L) for e in range(512, L + 256, 256)})
    ramp = torch.linspace(0.01, 1, 128, device=x.device)
    acc = torch.zeros(1, 6, 512, L, device=x.device)
    wsum = torch.zeros(1, 1, 1, L, device=x.device)
    for e in ends:
        tile = x[..., e - 512:e]
        y = model(tile.transpose(-1, -2)).transpose(-1, -2) if portrait else model(tile)
        w = torch.ones(1, 1, 1, 512, device=x.device)
        if e > 512:
            w[..., :128] = ramp
        if e < L:
            w[..., -128:] = ramp.flip(0)
        acc[..., e - 512:e] += y * w
        wsum[..., e - 512:e] += w
    out = (acc / wsum).clamp(0, 1)
    if portrait:
        out, x = out.transpose(-1, -2), x.transpose(-1, -2)
    # As test_large.py does, only what Flare7K++ removed is scaled up and taken
    # off the full-size frame, so the frame keeps its own resolution.
    removed = F.interpolate(x - out[:, :3], size=(rgb.height, rgb.width), mode="bilinear", align_corners=False)
    full = torch.from_numpy(np.asarray(rgb, np.float32) / 255).permute(2, 0, 1)[None]
    scene = (full - removed.cpu()).clamp(0, 1)[0]
    flare = F.interpolate(out[:, 3:], size=(rgb.height, rgb.width), mode="bilinear", align_corners=False)[0].cpu()
    # Flare strength: the flare layer's brightest channel
    return Image.fromarray((scene.permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)), flare.amax(0).numpy()


# ---------------------------------------------------------------- YOLO11-seg

_yolo = None


def _load_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(str(YOLO_WEIGHTS))
    return _yolo


def detect_objects(img: Image.Image, conf: float = 0.25) -> list[tuple[str, float, np.ndarray]]:
    """People and vehicles: (class name, confidence, [H, W] bool mask)."""
    rgb = img.convert("RGB")
    r = _load_yolo()(np.asarray(rgb)[:, :, ::-1], conf=conf, verbose=False, retina_masks=True)[0]
    found = []
    if r.masks is None:
        return found
    for cls, p, m in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.masks.data.cpu().numpy()):
        name = r.names[int(cls)]
        if name in PROTECTED:
            m = m > 0.5
            if m.shape != (rgb.height, rgb.width):
                m = cv2.resize(m.astype(np.uint8), (rgb.width, rgb.height), interpolation=cv2.INTER_NEAREST) > 0
            found.append((name, float(p), m))
    return found


# ---------------------------------------------------------------- SAM 3

_sam3 = None


def _load_sam3():
    global _sam3
    if _sam3 is None:
        from ultralytics.models.sam import SAM3SemanticPredictor
        _sam3 = SAM3SemanticPredictor(overrides=dict(conf=min(SAM3_PROMPTS.values()), task="segment", mode="predict",
                                                     model=str(SAM3_WEIGHTS), half=False, save=False, verbose=False))
    return _sam3


def find_concepts(img: Image.Image) -> list[tuple[str, float, np.ndarray]]:
    """SAM 3 on SAM3_PROMPTS: (prompt, confidence, [H, W] bool mask) for each
    instance at or above its prompt's confidence."""
    rgb = img.convert("RGB")
    pred = _load_sam3()
    pred.set_image(np.asarray(rgb)[:, :, ::-1].copy())
    prompts = list(SAM3_PROMPTS)
    r = pred(text=prompts)[0]
    found = []
    if r.masks is None:
        return found
    for m, c, p in zip(r.masks.data.cpu().numpy(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        name = prompts[int(c)]
        if p >= SAM3_PROMPTS[name]:
            m = m > 0.5
            if m.shape != (rgb.height, rgb.width):
                m = cv2.resize(m.astype(np.uint8), (rgb.width, rgb.height), interpolation=cv2.INTER_NEAREST) > 0
            found.append((name, float(p), m))
    return found


# ---------------------------------------------------------------- instances

def _components(strength: np.ndarray, threshold: float, min_area: int, grow: int) -> list[np.ndarray]:
    """Connected regions of strength > threshold, each grown by `grow` px so the
    soft edge is included; regions smaller than min_area px are dropped."""
    binary = (strength > threshold).astype(np.uint8)
    if grow:
        binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1  # largest first
    return [labels == i for i in order if stats[i, cv2.CC_STAT_AREA] >= min_area]


STEPS = ["Find glare (Flare7K++)", "Find raindrops and streaks (our DeRaindrop)",
         "Find people and vehicles (YOLO11-seg)", "Split into instances"]
SAM3_STEP = "Find the wet road and lens glare by name (SAM 3)"


def steps(use_sam3: bool = False) -> list[str]:
    return STEPS[:3] + [SAM3_STEP] + STEPS[3:] if use_sam3 else STEPS


def analyse(img: Image.Image, **kwargs) -> Analysis:
    """Find every glare, streak, raindrop and object instance in the frame
    (see analyse_steps for the settings)."""
    for _, _, a in analyse_steps(img, **kwargs):
        pass
    return a


def analyse_steps(img: Image.Image, glare_threshold: float = 0.2, glare_relative: float = 0.5,
                  rain_threshold: float = 0.7, streak_area: float = 0.004, streak_contrast: float = 0.06,
                  object_conf: float = 0.25, use_sam3: bool = False, road_share: float = 0.8):
    """analyse() one model at a time, for the demo: yields (step index into
    steps(use_sam3), seconds, Analysis so far) as each step finishes.

    Glare: flare layer above glare_threshold and above glare_relative x the
    frame's strongest flare, so only the real bloom around each light counts,
    not the faint haze Flare7K++ spreads over a whole night frame.
    Drops: raindrop map above rain_threshold, outside the glare. A region
    larger than streak_area of the frame is a streak if it is at least
    streak_contrast brighter than a 15 px ring around it.
    SAM 3: a glare / lens-flare instance with more than road_share of it on the
    wet road is a road reflection and stays part of the protected road."""
    rgb = img.convert("RGB")
    a = Analysis(rgb)
    px = rgb.width * rgb.height

    t = time.perf_counter()
    a.flare_free, a.flare_layer = flare7kpp(rgb)
    a.seconds["glare (Flare7K++)"] = time.perf_counter() - t
    yield 0, a.seconds["glare (Flare7K++)"], a

    t = time.perf_counter()
    a.derained, att = derainnet.derain(rgb, derainnet.FINETUNED)
    a.rain_map = att[0, 0].numpy()
    a.seconds["raindrops (our DeRaindrop)"] = time.perf_counter() - t
    yield 1, a.seconds["raindrops (our DeRaindrop)"], a

    t = time.perf_counter()
    objects = detect_objects(rgb, object_conf)
    a.seconds["objects (YOLO11-seg)"] = time.perf_counter() - t
    yield 2, a.seconds["objects (YOLO11-seg)"], a

    concepts = []
    if use_sam3:
        t = time.perf_counter()
        concepts = find_concepts(rgb)
        a.seconds["wet road and lens glare (SAM 3)"] = time.perf_counter() - t
        yield 3, a.seconds["wet road and lens glare (SAM 3)"], a

    t = time.perf_counter()
    glare_t = max(glare_threshold, glare_relative * float(a.flare_layer.max()))
    glare = np.zeros(a.flare_layer.shape, bool)
    for i, m in enumerate(_components(a.flare_layer, glare_t, px // 2000, grow=6), 1):
        a.instances.append(Instance(0, "glare", f"glare #{i}", m, float(a.flare_layer[m].mean())))
        glare |= m
    road = np.zeros(glare.shape, bool)
    for name, p, m in concepts:
        if name == "wet road":
            road |= m
    lens = [(p, m) for name, p, m in concepts if name != "wet road" and (m & road).sum() <= road_share * m.sum()]
    for p, m in lens:
        road &= ~m  # on the lens, not part of the scene
    # Lens glare Flare7K++ missed: its glare-free frame barely changes there
    # (e.g. lit web strands), so these are dimmed like streaks instead.
    sam_lens, n_lens = np.zeros(glare.shape, bool), 0
    for p, m in sorted(lens, key=lambda pm: -pm[0]):
        new = m & ~glare & ~sam_lens
        if new.sum() >= max(px // 2000, m.sum() // 2):
            n_lens += 1
            a.instances.append(Instance(0, "streak", f"lens glare #{n_lens} (SAM 3)", new, p))
            sam_lens |= new
    road &= ~glare
    if road.any():
        a.instances.append(Instance(0, "road", "wet road (kept)", road,
                                    max(p for name, p, _ in concepts if name == "wet road")))
    grey = np.asarray(rgb.convert("L"), np.float32) / 255
    n = {"drop": 0, "streak": 0}
    taken = glare | road | sam_lens
    for m in _components(np.where(taken, 0, a.rain_map), rain_threshold, px // 20000, grow=3):
        m &= ~taken
        kind = "drop"
        if m.sum() > streak_area * px:
            ring = cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~m
            if grey[m].mean() - grey[ring].mean() >= streak_contrast:
                kind = "streak"
        n[kind] += 1
        a.instances.append(Instance(0, kind, f"{kind} #{n[kind]}", m, float(a.rain_map[m].mean())))
    counts: dict[str, int] = {}
    for name, p, m in objects:
        counts[name] = counts.get(name, 0) + 1
        a.instances.append(Instance(0, "object", f"{name} #{counts[name]}", m, p))
    for i, inst in enumerate(a.instances, 1):
        inst.id = i
    yield len(steps(use_sam3)) - 1, time.perf_counter() - t, a


# ---------------------------------------------------------------- overlay

def _font(size: int):
    for f in ("C:/Windows/Fonts/segoeuib.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def overlay(a: Analysis, max_labels: int = 25) -> Image.Image:
    """The frame with every instance tinted and outlined in its kind's colour.
    The largest instances are also labelled (drops can number in the hundreds)."""
    base = np.asarray(a.image, np.float32)
    out = base.copy()
    for inst in sorted(a.instances, key=lambda i: -i.area):
        c = np.array(COLOURS[inst.kind], np.float32)
        alpha = 0.2 if inst.kind == "road" else 0.4  # the road is large: a light tint
        out[inst.mask] = (1 - alpha) * out[inst.mask] + alpha * c
    out = out.astype(np.uint8)
    for inst in a.instances:
        cs, _ = cv2.findContours(inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, COLOURS[inst.kind], 2)
    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    font = _font(max(14, a.image.height // 45))
    labelled = sorted(a.instances, key=lambda i: (i.kind == "drop", -i.area))[:max_labels]
    for inst in labelled:
        x0, y0, _, _ = inst.bbox
        tb = d.textbbox((x0, y0), inst.label, font=font)
        y = max(0, y0 - (tb[3] - tb[1]) - 6)
        d.rectangle((x0, y, x0 + tb[2] - tb[0] + 8, y + tb[3] - tb[1] + 6), fill=COLOURS[inst.kind])
        d.text((x0 + 4, y + 1 - (tb[1] - y0)), inst.label, font=font, fill=(15, 20, 30))
    return im


# ---------------------------------------------------------------- repair

REPAIRS = {
    "glare": "Flare7K++ glare removal",
    "drop": "our DeRaindrop restore",
    "streak": "softened toward the restored scene",
    "object": "protected: never softened or repainted",
    "road": "kept: real scene, no drop or streak repair",
}
# repair(whole_frame=True): DeRaindrop restores the whole frame first
REPAIRS_HYBRID = REPAIRS | {
    "drop": "our DeRaindrop restore (applied to the whole frame)",
    "glare": "Flare7K++ glare removal, on top of the restore",
    "object": "restored, but never softened",
    "road": "kept as the original frame: real scene",
}
FEATHER = {"glare": 8, "streak": 12, "drop": 5, "road": 12}  # px of soft edge


def _soft(masks: list[np.ndarray], shape: tuple[int, int], feather: int) -> np.ndarray:
    """The masks' union as a [H, W, 1] weight: 1 everywhere inside, fading to 0
    over `feather` px outside, so the repair reaches every pixel of an instance."""
    if not masks:
        return np.zeros((*shape, 1), np.float32)
    m = np.any(masks, 0).astype(np.uint8)
    if feather:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather + 1, 2 * feather + 1)))
        return np.maximum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), feather / 2), np.any(masks, 0))[..., None]
    return m.astype(np.float32)[..., None]


def repair(a: Analysis, whole_frame: bool = True) -> Image.Image:
    """Repair each instance with the model that suits it, then blend.

    whole_frame=True (the hybrid, default): start from our DeRaindrop's restore
    of the whole frame, since on real rain photos it also fixes the many faint
    drops too weak to be found as instances (58 real photos: 30.1 dB whole
    frame vs 28.1 dB repairing found drops only). Then:
      streaks -> dimmed further toward the darker restoration (below)
      glare   -> the glare Flare7K++ removes, taken off the restored frame
      road    -> back to the original frame (SAM 3's wet road: real scene)
    whole_frame=False: start from the original frame and repair only the instances:
      drops   -> our DeRaindrop's restored frame (it recovers what's behind the water)
      streaks -> whichever of DeRaindrop's and Flare7K++'s results is darker at
                 each pixel (all three colours from the same one, so no colour
                 fringes), so a streak can only be dimmed; skipped on people and vehicles
      glare   -> Flare7K++'s glare-free frame, applied last so it wins where they meet
    A black-and-white frame (night IR footage) stays black-and-white: the
    repairs' slight colour tints are dropped.
    Each kind's masks get a soft edge FEATHER px wide."""
    shape = (a.image.height, a.image.width)
    by_kind = {k: [i.mask for i in a.instances if i.kind == k] for k in COLOURS}
    names = REPAIRS_HYBRID if whole_frame else REPAIRS
    for inst in a.instances:
        inst.repair = names[inst.kind]
    x = np.asarray(a.image, np.float32)
    derained = np.asarray(a.derained, np.float32)
    flare_free = np.asarray(a.flare_free, np.float32)
    protected = _soft(by_kind["object"], shape, 3)
    darker = np.where((derained.sum(2) <= flare_free.sum(2))[..., None], derained, flare_free)
    w_streak = _soft(by_kind["streak"], shape, FEATHER["streak"]) * (1 - protected)
    # Each glare gets a soft edge in proportion to its size: a large bloom's
    # removal fades out over tens of pixels, so it leaves no dark rim.
    w_glare = np.zeros((*shape, 1), np.float32)
    for m in by_kind["glare"]:
        w_glare = np.maximum(w_glare, _soft([m], shape, max(FEATHER["glare"], round(0.12 * np.sqrt(m.sum())))))
    if whole_frame:
        out = derained.copy()
        out = out * (1 - w_streak) + np.minimum(darker, out) * w_streak
        out = out * (1 - w_glare) + np.clip(out - (x - flare_free), 0, 255) * w_glare
        w = _soft(by_kind["road"], shape, FEATHER["road"])
        out = out * (1 - w) + x * w
    else:
        out = x.copy()
        w = _soft(by_kind["drop"], shape, FEATHER["drop"])
        out = out * (1 - w) + derained * w
        out = out * (1 - w_streak) + np.minimum(darker, x) * w_streak
        out = out * (1 - w_glare) + flare_free * w_glare
    if np.abs(x - x.mean(2, keepdims=True)).mean() < 2:
        out = np.repeat(out.mean(2, keepdims=True), 3, 2)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def table(a: Analysis) -> list[list]:
    """One row per instance, for the demo: id, instance, type, % of frame, score, repair."""
    px = a.image.width * a.image.height
    return [[i.id, i.label, i.kind, round(100 * i.area / px, 2), round(i.score, 2), i.repair or REPAIRS_HYBRID[i.kind]]
            for i in a.instances]


def summary(a: Analysis) -> str:
    counts = {k: sum(i.kind == k for i in a.instances) for k in COLOURS}
    cover = {k: 100 * np.any([i.mask for i in a.instances if i.kind == k] or [np.zeros((1, 1), bool)], 0).mean()
             for k in COLOURS}
    times = ", ".join(f"{k} {v:.1f}s" for k, v in a.seconds.items())
    road = f", wet road kept ({cover['road']:.1f}%)" if counts["road"] else ""
    return (f"{counts['glare']} glare ({cover['glare']:.1f}% of frame), {counts['streak']} streaks "
            f"({cover['streak']:.1f}%), {counts['drop']} drops ({cover['drop']:.1f}%), "
            f"{counts['object']} objects ({cover['object']:.1f}%){road}  |  {times}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=RAINDROP_ROOT / "segment_repair_out")
    ap.add_argument("--half", action="store_true", help="process at half size (the CCTV screenshots are 2x)")
    ap.add_argument("--sam3", action="store_true", help="also use SAM 3 (wet road, lens glare)")
    args = ap.parse_args()
    paths = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) \
        if args.input.is_dir() else [args.input]
    args.out.mkdir(parents=True, exist_ok=True)
    for p in paths:
        img = Image.open(p).convert("RGB")
        if args.half:
            img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        a = analyse(img, use_sam3=args.sam3)
        name = "_".join(p.relative_to(args.input).with_suffix("").parts) if args.input.is_dir() else p.stem
        overlay(a).save(args.out / f"{name}_instances.jpg", quality=90)
        repair(a).save(args.out / f"{name}_repaired.png")
        print(f"{name}: {summary(a)}", flush=True)


if __name__ == "__main__":
    main()
