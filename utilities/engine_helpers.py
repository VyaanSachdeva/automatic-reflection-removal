"""
Helpers for the training Engine: compositing, metrics formatting, image conversion,
visualization logging, tensor cleanup, checkpoint loading, and evaluation metrics.
"""
import os
from typing import Any, Callable, Dict, Optional

import pandas as pd
import torch
import torchvision.transforms as transforms
import wandb

from metrics import mse_metric, psnr_metric, ssim_metric, luminance_suppression_ratio


def composite_specular_diffuse(
    specular: torch.Tensor, diffuse: torch.Tensor
) -> torch.Tensor:
    """
    Composite specular and diffuse components into a reconstructed image.

    Args:
        specular: [B, C, H, W] or [B, 4, H, W] for RGBA
        diffuse: [B, C, H, W] or [B, 4, H, W] for RGBA

    Returns:
        Reconstructed RGB [B, 3, H, W]
    """
    if specular.shape[1] == 4 and diffuse.shape[1] == 4:
        spec_rgb = specular[:, :3]
        spec_alpha = specular[:, 3:4]
        diff_rgb = diffuse[:, :3]
        diff_alpha = diffuse[:, 3:4]
        recon_rgb = spec_rgb * spec_alpha + diff_rgb * diff_alpha * (1 - spec_alpha)
        recon_rgb = torch.clamp(recon_rgb, 0, 1)
        return recon_rgb
    recon = specular + diffuse
    recon = recon / recon.max()
    return torch.clamp(recon, 0, 1)


def prepare_metrics_for_wandb(metrics: dict, phase: str) -> dict:
    """Format metrics dict with phase prefix for wandb logging."""
    formatted = {}
    for k, v in metrics.items():
        if "Step" in k:
            formatted[k] = v
        else:
            formatted[f"{phase}/{k}"] = v
    return formatted


def compute_eval_metrics(
    pred_decomposition: dict,
    gt_decomposition: dict,
    phase: str,
    pixel_supervision_mask: Optional[torch.Tensor] = None,
    pixel_inpaint_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute PSNR/SSIM/MSE for diffuse, specular, and rgb_highlighted (vectorized over batch).

    Args:
        pred_decomposition: Model output dict (diffuse, specular, rgb_highlighted).
        gt_decomposition: Ground truth dict with same keys.
        phase: "Training", "Validation", or "Test". Supervision mask used only when "Training".
        pixel_supervision_mask: Optional [B, 1, H, W] mask for diffuse metrics.
        pixel_inpaint_mask: Optional [B, 1, H, W] highlight/repair-region mask. Used to log
            IN-MASK diffuse metrics in EVERY phase. The default diffuse metrics above are
            computed over the supervision-complement (everything EXCEPT highlights), so they
            are blind to whether the highlight was removed and are gameable by an identity
            map; the in-mask metrics restrict to the region the model actually repairs and
            are the faithful comparison/selection signal.

    Returns:
        Dict of metric names to float values (e.g. "PSNR/diffuse", "PSNR/diffuse_inmask").
    """
    out: Dict[str, float] = {}
    eval_mask = pixel_supervision_mask if phase == "Training" else None
    try:
        if "diffuse" in pred_decomposition and "diffuse" in gt_decomposition:
            pdiff = pred_decomposition["diffuse"].detach()
            gt = gt_decomposition["diffuse"].detach()
            out["PSNR/diffuse"] = float(
                psnr_metric(pdiff, gt, mask=eval_mask, reduction="mean").item()
            )
            out["SSIM/diffuse"] = float(
                ssim_metric(pdiff, gt, mask=eval_mask, reduction="mean").item()
            )
            out["MSE/diffuse"] = float(
                mse_metric(pdiff, gt, mask=eval_mask, reduction="mean").item()
            )
            # IN-MASK (highlight repair region) diffuse metrics — logged for all phases.
            # NOTE: gt diffuse still contains REAL highlights, so these are most faithful
            # over the SYNTHETIC-highlight portion; evaluate on a fixed held-out synthetic
            # distribution for cross-run comparison. HighlightSuppression below is GT-free.
            if pixel_inpaint_mask is not None:
                im = pixel_inpaint_mask.detach().float()
                out["PSNR/diffuse_inmask"] = float(
                    psnr_metric(pdiff, gt, mask=im, reduction="mean").item()
                )
                out["SSIM/diffuse_inmask"] = float(
                    ssim_metric(pdiff, gt, mask=im, reduction="mean").item()
                )
                out["MSE/diffuse_inmask"] = float(
                    mse_metric(pdiff, gt, mask=im, reduction="mean").item()
                )
                if "rgb_highlighted" in gt_decomposition:
                    # Fraction of luminance removed inside the mask (input -> diffuse).
                    # Higher = stronger highlight removal. No clean GT required.
                    out["HighlightSuppression/diffuse_inmask"] = float(
                        luminance_suppression_ratio(
                            gt_decomposition["rgb_highlighted"].detach(),
                            pdiff,
                            im,
                            reduction="mean",
                        ).item()
                    )
        if "specular" in pred_decomposition and "specular" in gt_decomposition:
            ps = pred_decomposition["specular"].detach()
            gs = gt_decomposition["specular"].detach()
            out["PSNR/specular"] = float(psnr_metric(ps, gs, reduction="mean").item())
            out["SSIM/specular"] = float(ssim_metric(ps, gs, reduction="mean").item())
            out["MSE/specular"] = float(mse_metric(ps, gs, reduction="mean").item())
        if (
            "rgb_highlighted" in pred_decomposition
            and "rgb_highlighted" in gt_decomposition
        ):
            pr = pred_decomposition["rgb_highlighted"].detach()
            gr = gt_decomposition["rgb_highlighted"].detach()
            out["PSNR/recon"] = float(psnr_metric(pr, gr, reduction="mean").item())
            out["SSIM/recon"] = float(ssim_metric(pr, gr, reduction="mean").item())
            out["MSE/recon"] = float(mse_metric(pr, gr, reduction="mean").item())
    except Exception:
        pass
    return out


def to_cpu_image(tensor: Optional[torch.Tensor], batch_idx: int = 0):
    """Convert tensor to PIL Image; handles [B,C,H,W], [C,H,W], [H,W]."""
    if tensor is None:
        return None
    if tensor.dim() == 4:
        tensor = tensor[0].clone()
    elif tensor.dim() == 3:
        tensor = tensor.clone()
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).clone()
    tensor = tensor.cpu().detach().clamp(0, 1)
    pil_image = transforms.ToPILImage()(tensor)
    del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pil_image


def add_image_safely(
    viz_dict: dict,
    key: str,
    tensor: Optional[torch.Tensor],
    caption: str,
    batch_idx: int = 0,
    phase: Optional[str] = None,
    test_idx: Optional[int] = None,
    metrics_test_df: Optional[pd.DataFrame] = None,
) -> None:
    """Add image to viz dict and log to wandb with phase/test_idx prefixing."""
    if tensor is None:
        return
    image = wandb.Image(to_cpu_image(tensor, batch_idx))
    log_key = key
    if isinstance(phase, str):
        if phase == "Test" and test_idx is not None:
            if key.startswith("Test/") and not key.startswith(f"Test/test_idx_{test_idx}/"):
                log_key = key.replace("Test/", f"Test/test_idx_{test_idx}/", 1)
            elif not key.startswith(f"Test/test_idx_{test_idx}/"):
                log_key = f"Test/test_idx_{test_idx}/{key}"
        elif not key.startswith(("Training/", "Validation/", "Test/")):
            log_key = f"{phase}/{key}"
    viz_dict[log_key] = image
    payload = {log_key: image}
    if phase == "Test" and test_idx is not None:
        payload["Step/test_idx"] = int(test_idx)
    elif (
        metrics_test_df is not None
        and not metrics_test_df.empty
        and "Step/test_idx" in metrics_test_df.columns
    ):
        try:
            payload["Step/test_idx"] = int(metrics_test_df["Step/test_idx"].iloc[-1])
        except Exception:
            pass
    wandb.log(payload)


def cleanup_tensor_dict(
    tensor_dict: Optional[dict], keys_to_keep: Optional[list] = None
) -> None:
    """In-place cleanup of tensor dict; optionally keep specified keys."""
    if tensor_dict is None:
        return
    keys_to_keep = keys_to_keep or []
    to_delete = [k for k in tensor_dict.keys() if k not in keys_to_keep]
    for key in to_delete:
        if isinstance(tensor_dict[key], torch.Tensor):
            tensor_dict[key].detach_()
            if tensor_dict[key].is_cuda:
                tensor_dict[key].cpu()
            del tensor_dict[key]
        elif isinstance(tensor_dict[key], dict):
            cleanup_tensor_dict(tensor_dict[key])
            del tensor_dict[key]


def create_model_from_checkpoint(
    checkpoint_path: str, device: Optional[torch.device] = None
) -> Optional[dict]:
    """
    Load checkpoint and return model + state dicts. Does not build model class;
    returns dict with model_state_dict, optimizer_state_dict, config, epoch, etc.
    """
    if not os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".pt"):
            alt = checkpoint_path[:-3] + ".pth"
        elif checkpoint_path.endswith(".pth"):
            alt = checkpoint_path[:-4] + ".pt"
        else:
            return None
        if os.path.exists(alt):
            checkpoint_path = alt
        else:
            return None

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model_class_name = checkpoint.get("model_class_name")
    model_class_module = checkpoint.get("model_class_module")

    if not (model_class_name and model_class_module):
        return None

    import importlib
    module = importlib.import_module(model_class_module)
    model_class = getattr(module, model_class_name)
    model = model_class()
    model.load_state_dict(checkpoint["model_state_dict"])
    if device:
        model = model.to(device)

    return {
        "model": model,
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
        "LRscheduler_state_dict": checkpoint.get("LRscheduler_state_dict"),
        "LRschedulerPlateau_state_dict": checkpoint.get("LRschedulerPlateau_state_dict"),
        "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
        "epoch": checkpoint.get("epoch"),
        "config": checkpoint.get("config"),
        "model_class_name": model_class_name,
        "model_class_module": model_class_module,
    }
