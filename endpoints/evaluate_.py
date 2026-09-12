"""Evaluation metrics API for UnReflectAnything."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from os import PathLike

if False:
    from torch import Tensor


def evaluate(
    output: Union[str, PathLike, Path, "Tensor"],
    reference: Union[str, PathLike, Path, "Tensor"],
    metrics: Optional[List[str]] = None,
    mask: Optional[Union[str, PathLike, Path, "Tensor"]] = None,
) -> Dict[str, float]:
    """Compute evaluation metrics between output and reference images.

    This function computes image quality metrics comparing model outputs
    to reference (ground truth) images. It supports both file-based and
    tensor-based inputs.

    Args:
        output: Model output to evaluate. Can be:
            - Path to a single image file
            - Path to a directory of images
            - Tensor of shape [B, C, H, W]
        reference: Reference (ground truth) images. Same format as output.
        metrics: List of metrics to compute. If None, computes all available.
        mask: Optional mask for masked evaluation.

    Returns:
        Dictionary mapping metric names to their values.
    """
    from evaluate import evaluate_images

    return evaluate_images(
        output=output,
        reference=reference,
        metrics=metrics,
        mask=mask,
    )
