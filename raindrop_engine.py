"""
RaindropEngine: the training engine with VRGS (raindrop_render.py's
RaindropRender) as its degradation renderer, in place of UnReflectAnything's
specular-highlight synthesis. Everything else in Engine is reused unchanged:
loss computation, optimizer, checkpointing, logging, the token inpainter and
the decoders. VRGS is lens-relative, so unlike the original synthesis it needs
no depth model (MoGe).
"""
from __future__ import annotations

import random

import torch

from engine import Engine
from raindrop_render import RaindropRender
from utilities.model import pixel_mask_to_patch_mask


class RaindropEngine(Engine):
    def _build_degradation_renderer(self) -> torch.nn.Module:
        return RaindropRender(height=self.height, width=self.width).to(self.device)

    def _synthesize_degradation(self, sample, is_training: bool, batch_idx: int) -> dict:
        rgb = sample["diffuse"].to(self.device, non_blocking=True)

        # During validation/test, freeze rain intensity the same way
        # HighlightRender freezes highlight params -- otherwise a harder-rain
        # eval batch isn't comparable across epochs/runs.
        freeze_val = (not is_training) and bool(self.config.get("FREEZE_VAL_RAIN", False))
        if freeze_val:
            intensity_cfg = self.config.get("VAL_RAIN_INTENSITY", 0.6)
            _seed = int(self.config.get("VAL_RAIN_SEED", 1234)) + batch_idx
            _cpu_rng_state = torch.get_rng_state()
            _py_rng_state = random.getstate()
            random.seed(_seed)
            torch.manual_seed(_seed)
        else:
            intensity_cfg = self.config.get("RAIN_INTENSITY", [0.2, 1.0])

        if isinstance(intensity_cfg, (list, tuple)) and len(intensity_cfg) == 2:
            rain_intensity = random.uniform(float(intensity_cfg[0]), float(intensity_cfg[1]))
        else:
            rain_intensity = float(intensity_cfg)

        capture_context = self.config.get("CAPTURE_CONTEXT", "dashcam")
        result = self.add_highlights(rgb, rain_intensity=rain_intensity, capture_context=capture_context)

        if freeze_val:
            torch.set_rng_state(_cpu_rng_state)
            random.setstate(_py_rng_state)

        degraded = result["degraded"]
        pixel_inpaint_soft_mask = result["mask"]  # [B,1,H,W]

        threshold = float(self.config.get("DATASET_HIGHLIGHT_SUPERVISION_THRESHOLD", 0.1))
        pixel_inpaint_mask = (pixel_inpaint_soft_mask > threshold).bool()
        patch_inpaint_mask = pixel_mask_to_patch_mask(
            pixel_inpaint_mask, patch_size=16, threshold=threshold, invert=False
        )

        # No pre-existing degradation in the clean base pool to exclude from
        # supervision (unlike HighlightRender's dataset-highlight handling for
        # real photos that may already contain highlights) -- supervise
        # everywhere. Bool, not float: patch_supervision_mask is used for
        # tensor indexing downstream (loss_utils.py), which requires an
        # integer/bool index tensor.
        pixel_supervision_mask = torch.ones_like(pixel_inpaint_mask, dtype=torch.bool)
        patch_supervision_mask = torch.ones_like(patch_inpaint_mask, dtype=torch.bool)

        return {
            "highlight_result": {"rgb_highlighted": degraded},
            "pixel_supervision_mask": pixel_supervision_mask,
            "patch_supervision_mask": patch_supervision_mask,
            "pixel_inpaint_soft_mask": pixel_inpaint_soft_mask,
            "pixel_inpaint_mask": pixel_inpaint_mask,
            "patch_inpaint_mask": patch_inpaint_mask,
        }
