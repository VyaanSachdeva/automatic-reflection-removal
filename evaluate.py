"""Evaluation module for computing image quality metrics.

This module provides functions to evaluate model outputs against reference images
using various image quality metrics. It wraps the metrics.py functions with a
high-level interface for both file-based and tensor-based evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
from torch import Tensor

if TYPE_CHECKING:
    from os import PathLike

# Available metrics and their properties
AVAILABLE_METRICS = {
    "psnr": {"function": "psnr_metric", "higher_better": True, "name": "PSNR (dB)"},
    "ssim": {"function": "ssim_metric", "higher_better": True, "name": "SSIM"},
    "mse": {"function": "mse_metric", "higher_better": False, "name": "MSE"},
    "deltaE2000": {
        "function": "deltaE2000_metric",
        "higher_better": False,
        "name": "Delta E 2000",
    },
    "gmsd": {"function": "gmsd_metric", "higher_better": False, "name": "GMSD"},
    "dists": {"function": "dists_metric", "higher_better": False, "name": "DISTS"},
}


def _load_image_as_tensor(path: Union[str, PathLike, Path]) -> Tensor:
    """Load an image file as a tensor [1, 3, H, W] with values in [0, 1]."""
    from PIL import Image
    from torchvision.transforms import functional as TF

    path = Path(path)
    with Image.open(path) as img:
        rgb_img = img.convert("RGB")
        tensor = TF.to_tensor(rgb_img)  # [3, H, W], values in [0, 1]
    return tensor.unsqueeze(0)  # [1, 3, H, W]


def _load_images_from_directory(
    directory: Path,
    extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
) -> tuple[Tensor, List[Path]]:
    """Load all images from a directory as a batched tensor.

    Args:
        directory: Path to the directory containing images.
        extensions: Tuple of valid image extensions.

    Returns:
        Tuple of (batched tensor [B, 3, H, W], list of paths).
    """
    from PIL import Image
    from torchvision.transforms import functional as TF

    paths = sorted(
        [
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]
    )

    if not paths:
        raise ValueError(f"No images found in {directory}")

    tensors = []
    for path in paths:
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            tensor = TF.to_tensor(rgb_img)
            tensors.append(tensor)

    # Stack tensors - they should all be the same size
    # If sizes differ, we'll raise an error
    try:
        batch = torch.stack(tensors, dim=0)  # [B, 3, H, W]
    except RuntimeError as e:
        raise ValueError(
            f"Images in {directory} have different sizes. "
            "All images must have the same dimensions for batch evaluation."
        ) from e

    return batch, paths


def _match_images(
    output_paths: List[Path],
    reference_paths: List[Path],
    output_root: Path,
    reference_root: Path,
) -> tuple[List[Path], List[Path]]:
    """Match output and reference images by relative path or filename.

    Args:
        output_paths: List of output image paths.
        reference_paths: List of reference image paths.
        output_root: Root directory of output images.
        reference_root: Root directory of reference images.

    Returns:
        Tuple of (matched output paths, matched reference paths).
    """
    # Build lookup by relative path
    ref_by_relative = {}
    ref_by_name = {}
    for ref_path in reference_paths:
        try:
            relative = ref_path.relative_to(reference_root)
            ref_by_relative[str(relative)] = ref_path
        except ValueError:
            pass
        ref_by_name[ref_path.name] = ref_path

    matched_output = []
    matched_reference = []

    for out_path in output_paths:
        # Try to match by relative path first
        try:
            relative = out_path.relative_to(output_root)
            if str(relative) in ref_by_relative:
                matched_output.append(out_path)
                matched_reference.append(ref_by_relative[str(relative)])
                continue
        except ValueError:
            pass

        # Try to match by filename
        if out_path.name in ref_by_name:
            matched_output.append(out_path)
            matched_reference.append(ref_by_name[out_path.name])

    if not matched_output:
        raise ValueError(
            "No matching images found between output and reference directories. "
            "Images are matched by relative path or filename."
        )

    return matched_output, matched_reference


def evaluate_images(
    output: Union[str, PathLike, Path, Tensor],
    reference: Union[str, PathLike, Path, Tensor],
    metrics: Optional[List[str]] = None,
    mask: Optional[Union[str, PathLike, Path, Tensor]] = None,
) -> Dict[str, float]:
    """Compute evaluation metrics between output and reference images.

    This function computes image quality metrics comparing model outputs
    to reference (ground truth) images. It supports both file-based and
    tensor-based inputs.

    Args:
        output: Model output to evaluate. Can be:
            - Path to a single image file
            - Path to a directory of images
            - Tensor of shape [B, C, H, W] or [C, H, W]
        reference: Reference (ground truth) images. Same format as output.
        metrics: List of metrics to compute. If None, computes all available:
            - "psnr": Peak Signal-to-Noise Ratio (higher is better)
            - "ssim": Structural Similarity Index (higher is better)
            - "mse": Mean Squared Error (lower is better)
            - "deltaE2000": Color difference in LAB space (lower is better)
            - "gmsd": Gradient Magnitude Similarity Deviation (lower is better)
            - "dists": Deep Image Structure and Texture Similarity (lower is better)
        mask: Optional mask for masked evaluation. Same spatial size as images.

    Returns:
        Dictionary mapping metric names to their values.

    Raises:
        FileNotFoundError: If output or reference paths don't exist.
        ValueError: If output and reference have mismatched shapes/counts.

    Example:
        >>> # Evaluate directory of images
        >>> results = evaluate_images("outputs/", "references/")
        >>> print(f"PSNR: {results['psnr']:.2f} dB")

        >>> # Evaluate specific metrics on tensors
        >>> results = evaluate_images(pred_tensor, gt_tensor, metrics=["psnr", "ssim"])
    """
    import metrics as metrics_module

    # Determine which metrics to compute
    if metrics is None:
        metrics_to_compute = list(AVAILABLE_METRICS.keys())
    else:
        metrics_to_compute = [m.lower() for m in metrics]
        # Validate metrics
        for m in metrics_to_compute:
            if m not in AVAILABLE_METRICS:
                raise ValueError(
                    f"Unknown metric: {m}. Available: {list(AVAILABLE_METRICS.keys())}"
                )

    # Load output images
    if isinstance(output, Tensor):
        output_tensor = output
        if output_tensor.dim() == 3:
            output_tensor = output_tensor.unsqueeze(0)  # [1, C, H, W]
    else:
        output_path = Path(output).expanduser().resolve()
        if not output_path.exists():
            raise FileNotFoundError(f"Output path not found: {output_path}")

        if output_path.is_file():
            output_tensor = _load_image_as_tensor(output_path)
            output_paths = [output_path]
            is_single_file = True
        else:
            output_tensor, output_paths = _load_images_from_directory(output_path)
            is_single_file = False

    # Load reference images
    if isinstance(reference, Tensor):
        reference_tensor = reference
        if reference_tensor.dim() == 3:
            reference_tensor = reference_tensor.unsqueeze(0)  # [1, C, H, W]
    else:
        reference_path = Path(reference).expanduser().resolve()
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference path not found: {reference_path}")

        if reference_path.is_file():
            reference_tensor = _load_image_as_tensor(reference_path)
        else:
            # Match output and reference images
            if isinstance(output, Tensor):
                reference_tensor, _ = _load_images_from_directory(reference_path)
            else:
                ref_tensor_full, ref_paths_full = _load_images_from_directory(
                    reference_path
                )

                # Match images
                if is_single_file:
                    # Single output file - find matching reference
                    ref_by_name = {p.name: (i, p) for i, p in enumerate(ref_paths_full)}
                    out_name = output_paths[0].name
                    if out_name in ref_by_name:
                        idx, _ = ref_by_name[out_name]
                        reference_tensor = ref_tensor_full[idx : idx + 1]
                    else:
                        raise ValueError(f"No matching reference found for {out_name}")
                else:
                    # Directory - match by path/name
                    matched_out, matched_ref = _match_images(
                        output_paths, ref_paths_full, output_path, reference_path
                    )
                    # Reload matched images in order
                    out_tensors = [_load_image_as_tensor(p) for p in matched_out]
                    ref_tensors = [_load_image_as_tensor(p) for p in matched_ref]
                    output_tensor = torch.cat(out_tensors, dim=0)
                    reference_tensor = torch.cat(ref_tensors, dim=0)

    # Validate shapes
    if output_tensor.shape != reference_tensor.shape:
        raise ValueError(
            f"Shape mismatch: output {output_tensor.shape} vs reference {reference_tensor.shape}"
        )

    # Load mask if provided
    mask_tensor = None
    if mask is not None:
        if isinstance(mask, Tensor):
            mask_tensor = mask
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.unsqueeze(0)  # [1, 1, H, W] or [1, C, H, W]
        else:
            mask_path = Path(mask).expanduser().resolve()
            if mask_path.exists():
                mask_tensor = _load_image_as_tensor(mask_path)
                # Convert to single channel if needed
                if mask_tensor.shape[1] > 1:
                    mask_tensor = mask_tensor.mean(dim=1, keepdim=True)

    # Move to same device
    device = output_tensor.device
    reference_tensor = reference_tensor.to(device)
    if mask_tensor is not None:
        mask_tensor = mask_tensor.to(device)

    # Compute metrics
    results = {}

    for metric_name in metrics_to_compute:
        metric_info = AVAILABLE_METRICS[metric_name]
        func_name = metric_info["function"]
        func = getattr(metrics_module, func_name, None)

        if func is None:
            print(
                f"Warning: Metric function {func_name} not found, skipping {metric_name}"
            )
            continue

        try:
            # Some metrics don't support mask
            if metric_name in ("psnr",):
                # PSNR doesn't take mask in the current implementation
                value = func(output_tensor, reference_tensor, reduction="mean")
            elif mask_tensor is not None and metric_name in (
                "ssim",
                "mse",
                "deltaE2000",
            ):
                value = func(
                    output_tensor, reference_tensor, mask=mask_tensor, reduction="mean"
                )
            elif mask_tensor is not None and metric_name in ("gmsd", "dists"):
                value = func(
                    output_tensor, reference_tensor, mask=mask_tensor, reduction="mean"
                )
            else:
                value = func(output_tensor, reference_tensor, reduction="mean")

            # Convert to Python float
            if isinstance(value, Tensor):
                value = float(value.item())
            else:
                value = float(value)

            results[metric_name] = value

        except Exception as e:
            print(f"Warning: Failed to compute {metric_name}: {e}")
            results[metric_name] = float("nan")

    return results


def print_results(
    results: Dict[str, float],
    title: str = "Evaluation Results",
) -> None:
    """Print evaluation results in a formatted table.

    Args:
        results: Dictionary of metric names to values.
        title: Title for the results table.
    """
    print(f"\n{title}")
    print("-" * 40)
    for metric_name, value in results.items():
        info = AVAILABLE_METRICS.get(
            metric_name, {"name": metric_name, "higher_better": None}
        )
        display_name = info.get("name", metric_name)
        higher_better = info.get("higher_better")

        direction = ""
        if higher_better is True:
            direction = " (higher is better)"
        elif higher_better is False:
            direction = " (lower is better)"

        print(f"  {display_name}: {value:.4f}{direction}")
    print("-" * 40)


# Module exports
__all__ = [
    "evaluate_images",
    "print_results",
    "AVAILABLE_METRICS",
]
