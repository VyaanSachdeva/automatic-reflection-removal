"""Inference API for UnReflectAnything.

Thin wrapper around model loading + forward pass, usable both from the
Python API (``unreflectanything.inference(...)``) and from the CLI
(``unreflectanything inference ...``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from os import PathLike

import torch

__all__ = ["inference"]


def _config_to_path_or_none(
    config: Optional[Union[str, PathLike, Path, dict]],
) -> Optional[Path]:
    """Return a Path for model config when config is a path; else None (use model default)."""
    if config is None or isinstance(config, dict):
        return None
    p = Path(config).expanduser().resolve()
    return p if p.exists() else None


def _ensure_model(
    model: Optional[Any],
    weights_path: Optional[Union[str, PathLike, Path]],
    config: Optional[Union[str, PathLike, Path, dict]],
    device: str,
    verbose: bool,
) -> Any:
    """Return a ready-to-use ``UnReflectModel`` (creates one when *model* is ``None``)."""
    if model is not None:
        return model

    from ._shared import DEFAULT_WEIGHTS_FILENAME, get_cache_dir, _resolve_device
    from .model_ import model

    resolved_weights = (
        get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
        if weights_path is None
        else Path(weights_path).expanduser().resolve()
    )
    if not resolved_weights.exists():
        raise FileNotFoundError(
            f"Weights not found at {resolved_weights}. "
            "Run 'unreflectanything download --weights' first."
        )
    return model(
        pretrained=True,
        weights_path=resolved_weights,
        # config_path=_config_to_path_or_none(config),
        device=torch.device(_resolve_device(device)),
        verbose=verbose,
    )


def _model_device(model: Any, fallback: str = "cuda") -> torch.device:
    """Resolve the torch device the model lives on."""
    from ._shared import _resolve_device

    dev = getattr(model, "device", None)
    if dev is None:
        return torch.device(_resolve_device(fallback))
    return torch.device(dev) if isinstance(dev, str) else dev


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inference(
    input: Union[str, PathLike, Path, torch.Tensor],
    output: Optional[Union[str, PathLike, Path]] = None,
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    config: Optional[Union[str, PathLike, Path, dict]] = None,
    device: Optional[str] = None,
    batch_size: int = 4,
    threshold: float = 0.3,
    dilation: int = 40,
    resize_output: bool = True,
    verbose: bool = False,
    model: Optional[Any] = None,
    show_progress: bool = False,
    composite: bool = False,
) -> Optional[torch.Tensor]:
    """Run inference on input image(s) to remove specular reflections.

    Wraps model loading and forward pass.  Works with tensors (API) or
    file / directory paths (CLI / scripting).

    Args:
        input: Path to an image file or directory, **or** a ``[B, 3, H, W]``
            float tensor with values in ``[0, 1]``.
        output: When given, results are saved to this path and ``None`` is
            returned.  When ``None``, results are returned as a tensor.
        weights_path: Checkpoint path.  Ignored when *model* is provided.
        config: Architecture config (YAML path or dict).  Ignored when
            *model* is provided.
        device: Target device: ``"cuda"`` / ``"gpu"`` or ``"cpu"``.  If ``None``,
            auto-detected (CUDA if available, else CPU).  Ignored when *model* is provided.
        batch_size: Images per forward pass when processing directories.
        threshold: Highlight-mask threshold passed to the model (default 0.3).
        dilation: Highlight-mask dilation in pixels passed to the model
            (default 40).
        resize_output: Resize saved images back to original dimensions.
        verbose: Print progress information.
        model: Pre-built ``UnReflectModel`` instance.  When provided the
            function skips model creation (weights_path / config / device
            are ignored).
        show_progress: If ``True``, show a tqdm progress bar over images (used by CLI).

    Returns:
        ``[B, 3, H, W]`` diffuse tensor when *output* is ``None``, else
        ``None`` (results written to disk).
    """
    from torch import Tensor
    from ._shared import _resolve_device

    if device is None or (isinstance(device, str) and not device.strip()):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device.strip().lower()
        if device == "gpu":
            device = "cuda"
    device = _resolve_device(device)
    print(f"Using device: {device}")

    is_tensor_input = isinstance(input, Tensor)

    if is_tensor_input:
        return _inference_tensor(
            input_tensor=input,
            model=model,
            weights_path=weights_path,
            config=config,
            device=device,
            threshold=threshold,
            dilation=dilation,
            verbose=verbose,
            composite=composite,
        )

    input_path = Path(input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if output is None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            return _inference_files_return_tensors(
                input_path=input_path,
                output_path=Path(tmpdir),
                model=model,
                weights_path=weights_path,
                config=config,
                device=device,
                batch_size=batch_size,
                threshold=threshold,
                dilation=dilation,
                resize_output=resize_output,
                verbose=verbose,
                composite=composite,
            )
    else:
        output_path = Path(output).expanduser().resolve()
        # Single file in → allow output to be a file path; dir in → output is a dir
        from ._shared import DEFAULT_IMAGE_EXTENSIONS
        output_is_file = (
            input_path.is_file()
            and output_path.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
        )
        if output_is_file:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path.mkdir(parents=True, exist_ok=True)
        mdl = _ensure_model(model, weights_path, config, device, verbose)
        _inference_files_save(
            model=mdl,
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
            threshold=threshold,
            dilation=dilation,
            resize_output=resize_output,
            verbose=verbose,
            show_progress=show_progress,
            composite=composite,
        )
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _inference_tensor(
    input_tensor: torch.Tensor,
    model: Optional[Any] = None,
    weights_path: Optional[Union[str, Path]] = None,
    config: Optional[Union[str, Path, dict]] = None,
    device: str = "cuda",
    threshold: float = 0.3,
    dilation: int = 40,
    verbose: bool = False,
    composite: bool = False,
) -> torch.Tensor:
    """Run inference on a ``[B, 3, H, W]`` tensor, return ``[B, 3, H, W]`` diffuse."""
    if input_tensor.dim() != 4:
        raise ValueError(
            f"Input tensor must be 4D [B,C,H,W], got {input_tensor.dim()}D"
        )
    if input_tensor.shape[1] != 3:
        raise ValueError(
            f"Input tensor must have 3 channels, got {input_tensor.shape[1]}"
        )

    mdl = _ensure_model(model, weights_path, config, device, verbose)
    torch_device = _model_device(mdl, device)
    input_tensor = input_tensor.to(device=torch_device, dtype=torch.float32)  # [B,3,H,W]

    mdl.eval()
    with torch.no_grad():
        diffuse = mdl(
            images=input_tensor,
            threshold=threshold,
            dilation=dilation,
            composite=composite,
        )  # [B, 3, H, W]
    return diffuse


def _inference_files_return_tensors(
    input_path: Path,
    output_path: Path,
    model: Optional[Any] = None,
    weights_path: Optional[Union[str, Path]] = None,
    config: Optional[Union[str, Path, dict]] = None,
    device: str = "cuda",
    batch_size: int = 4,
    threshold: float = 0.3,
    dilation: int = 40,
    resize_output: bool = True,
    verbose: bool = False,
    composite: bool = False,
) -> torch.Tensor:
    """Load images from disk, run model forward, return stacked tensor."""
    from PIL import Image
    from torchvision.transforms import functional as TF

    from ._model_builder import list_image_paths_minimal
    from ._shared import DEFAULT_IMAGE_EXTENSIONS

    mdl = _ensure_model(model, weights_path, config, device, verbose)
    torch_device = _model_device(mdl, device)

    target_side = getattr(mdl, "image_size", None) or getattr(
        mdl._model.dinov3.config, "image_size", 448
    )
    target_size = (target_side, target_side)

    image_paths = (
        [input_path]
        if input_path.is_file()
        else list_image_paths_minimal(input_path, DEFAULT_IMAGE_EXTENSIONS, verbose=verbose)
    )

    results = []
    mdl.eval()

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        for path in batch_paths:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                tensor = TF.to_tensor(rgb_img)
                resized = TF.resize(tensor, target_size, antialias=True)
                batch_tensors.append(resized)

        rgb_batch = torch.stack(batch_tensors, dim=0).to(
            device=torch_device, dtype=torch.float32,
        )  # [B, 3, H, W]

        with torch.no_grad():
            diffuse = mdl(
                images=rgb_batch,
                threshold=threshold,
                dilation=dilation,
                composite=composite,
            )  # [B, 3, H, W]
        results.append(diffuse.cpu())

    return torch.cat(results, dim=0)  # [N, 3, H, W]


def _inference_files_save(
    model: Any,
    input_path: Path,
    output_path: Path,
    batch_size: int = 4,
    threshold: float = 0.3,
    dilation: int = 40,
    resize_output: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
    composite: bool = False,
) -> None:
    """Run model forward on image files and save results to disk."""
    from PIL import Image
    from torchvision.transforms import functional as TF

    from ._model_builder import list_image_paths_minimal, save_diffuse_batch_minimal
    from ._shared import DEFAULT_IMAGE_EXTENSIONS

    torch_device = _model_device(model)

    target_side = getattr(model, "image_size", None) or getattr(
        model._model.dinov3.config, "image_size", 448
    )
    target_size = (target_side, target_side)

    input_dir = input_path if input_path.is_dir() else input_path.parent
    # Single file out: output_path is the exact destination file; else output is a dir
    output_is_single_file = (
        input_path.is_file() and bool(output_path.suffix)
        and output_path.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
    )
    output_dir = (
        output_path.parent if output_is_single_file
        else (output_path if output_path.is_dir() else output_path.parent)
    )
    image_paths = (
        [input_path]
        if input_path.is_file()
        else list_image_paths_minimal(input_dir, DEFAULT_IMAGE_EXTENSIONS, verbose=verbose)
    )

    model.eval()

    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(image_paths), unit="img", desc="Inference")
    else:
        pbar = None

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        original_sizes = []
        for path in batch_paths:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                original_sizes.append(rgb_img.size[::-1])  # (H, W)
                tensor = TF.to_tensor(rgb_img)
                resized = TF.resize(tensor, target_size, antialias=True)
                batch_tensors.append(resized)

        rgb_batch = torch.stack(batch_tensors, dim=0).to(
            device=torch_device, dtype=torch.float32,
        )  # [B, 3, H, W]

        with torch.no_grad():
            diffuse = model(
                images=rgb_batch,
                threshold=threshold,
                dilation=dilation,
                composite=composite,
            )  # [B, 3, H, W]

        if output_is_single_file and len(batch_paths) == 1:
            # Save single image to the exact output file path
            out_tensor = diffuse[0].clamp(0.0, 1.0).cpu()
            if resize_output and original_sizes:
                out_tensor = TF.resize(out_tensor, original_sizes[0], antialias=True)
            TF.to_pil_image(out_tensor).save(output_path)
        else:
            save_diffuse_batch_minimal(
                diffuse,
                batch_paths,
                input_dir,
                output_dir,
                original_sizes=original_sizes if resize_output else None,
                resize_output=resize_output,
            )

        if pbar is not None:
            pbar.update(len(batch_paths))

    if pbar is not None:
        pbar.close()


# ---------------------------------------------------------------------------
# Legacy CLI entry-point (YAML config → InferenceOptions)
# ---------------------------------------------------------------------------


def parse_cli():
    """Parse command line arguments and YAML file into inference options."""
    import yaml
    import argparse
    from inference import InferenceOptions

    parser = argparse.ArgumentParser(
        description="Run UnReflectAnything diffuse inference"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="./config/inference.yaml",
        required=False,
        help="Absolute path to the inference YAML options file (default: ./config_inference.yaml)",
    )

    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading inference configuration from '{config_path}'")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_options = yaml.safe_load(handle)

    def _as_path(value: Optional[str]) -> Optional[Path]:
        return None if value is None else Path(value).expanduser().resolve()

    raw_weights = raw_options.get("weights_path")
    weights_path = _as_path(raw_weights)
    if weights_path is None or (
        isinstance(raw_weights, str) and str(raw_weights).strip().lower() == "default"
    ):
        try:
            from ._shared import (
                DEFAULT_WEIGHTS_FILENAME,
                get_cache_dir,
            )

            weights_path = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
        except ImportError:
            weights_path = None
    input_dir = _as_path(raw_options.get("input_dir"))
    output_dir = _as_path(raw_options.get("output_dir"))

    if weights_path is None or not weights_path.exists():
        cache_dir = None
        try:
            from ._shared import get_cache_dir

            cache_dir = get_cache_dir("weights")
        except ImportError:
            pass
        hint = (
            " Run 'unreflectanything download-weights' first, or set weights_path in the config."
            if cache_dir
            else ""
        )
        raise FileNotFoundError(
            f"weights_path must point to an existing checkpoint file.{hint}"
        )
    if input_dir is None or not input_dir.exists():
        raise FileNotFoundError("input_dir must point to an existing directory")
    if output_dir is None:
        raise ValueError("output_dir must be provided")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[SUCCESS]  Configuration loaded",
    )

    batch_size = int(raw_options.get("batch_size", 4))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    image_extensions = raw_options.get("image_extensions")
    if image_extensions is None:
        extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        )
    else:
        extensions = tuple(ext.lower() for ext in image_extensions)

    resize_output = raw_options.get("resize_output", True)
    if not isinstance(resize_output, bool):
        raise ValueError("resize_output must be a boolean")

    # Accept both new names (threshold / dilation) and legacy names
    brightness_threshold = float(
        raw_options.get(
            "threshold",
            raw_options.get("brightness_threshold", 0.3),
        )
    )
    if not (0.0 <= brightness_threshold <= 1.0):
        raise ValueError("threshold must be between 0.0 and 1.0")

    inpaint_mask_dilation = int(
        raw_options.get(
            "dilation",
            raw_options.get("inpaint_mask_dilation", 40),
        )
    )

    monitor_usage = raw_options.get("monitor_usage", False)
    if not isinstance(monitor_usage, bool):
        raise ValueError("monitor_usage must be a boolean")

    num_workers = int(raw_options.get("num_workers", 4))
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    raw_composite = raw_options.get("composite", raw_options.get("COMPOSITE_AT_INFERENCE", False))
    if isinstance(raw_composite, dict):
        raw_composite = raw_composite.get("value", False)
    composite = bool(raw_composite)

    options = InferenceOptions(
        weights_path=weights_path,
        input_dir=input_dir,
        output_dir=output_dir,
        inpaint_mask_dilation=inpaint_mask_dilation,
        run=raw_options.get("run"),
        runs_dir=_as_path(raw_options.get("runs_dir")),
        model_config_path=_as_path(raw_options.get("model_config_path")),
        model_module=raw_options.get("model_module"),
        batch_size=batch_size,
        device=raw_options.get("device", "cuda"),
        image_extensions=extensions,
        resize_output=resize_output,
        brightness_threshold=brightness_threshold,
        monitor_usage=monitor_usage,
        num_workers=num_workers,
        composite=composite,
    )
    return options
