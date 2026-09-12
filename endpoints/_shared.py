"""Shared utilities and constants for UnReflectAnything endpoints.

Provides: cache dir, default weights filename, download helpers from HF,
image extensions, path collection, device resolution, and config application
for inference options.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

# Default Hugging Face repo for assets (override with UNREFLECTANYTHING_WEIGHTS_REPO)
DEFAULT_WEIGHTS_REPO = "AlbeRota/UnReflectAnything"

# Default checkpoint filename under the cache dir (must match what the HF repo provides)
DEFAULT_WEIGHTS_FILENAME = "full_model_weights.pt"
DEFAULT_CONFIG_FILENAME = "pretrained_config.yaml"

# Subfolder names in the HuggingFace repo (must match mgmt/upload_hf.py)
WEIGHTS_SUBFOLDER = "weights"
IMAGES_SUBFOLDER = "sample_images"
NOTEBOOKS_SUBFOLDER = "notebooks"
CONFIGS_SUBFOLDER = "configs"

# Default image extensions (consistent with inference.py)
DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


from .cache_ import get_cache_dir  # noqa: F811 — canonical impl lives in cache_.py


def _ensure_huggingface_hub():
    """Ensure huggingface_hub is installed, exit with helpful message if not."""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download

        return snapshot_download, hf_hub_download
    except ImportError:
        print(
            "The 'huggingface_hub' package is required for downloading assets. "
            "Install it with: pip install unreflectanything[weights]",
            file=sys.stderr,
        )
        sys.exit(1)


def download_weights(
    output_dir: Optional[Path] = None,
    variant: str = "default",
    force: bool = False,
) -> Path:
    """Download pretrained weights to the given directory."""
    snapshot_download, _ = _ensure_huggingface_hub()
    repo_id = os.environ.get("UNREFLECTANYTHING_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO)
    if output_dir is None:
        output_dir = get_cache_dir(subdir="weights")
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = None if variant == "default" else variant
    # Only download .pt weight files; skip blobs/ and other repo internals
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        revision=revision,
        force_download=force,
        allow_patterns=[f"{WEIGHTS_SUBFOLDER}/*.pt"],
    )
    subfolder_path = output_dir / WEIGHTS_SUBFOLDER
    if subfolder_path.exists() and subfolder_path.is_dir():
        for item in subfolder_path.iterdir():
            target = output_dir / item.name
            if not target.exists() or force:
                item.rename(target)
        try:
            subfolder_path.rmdir()
        except OSError:
            pass
    print(f"Weights saved to {output_dir}")
    return output_dir


def download_images(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download sample images for testing the model."""
    snapshot_download, _ = _ensure_huggingface_hub()
    repo_id = os.environ.get("UNREFLECTANYTHING_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO)
    if output_dir is None:
        output_dir = get_cache_dir(subdir="images")
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        force_download=force,
        allow_patterns=[f"{IMAGES_SUBFOLDER}/*"],
    )
    subfolder_path = output_dir / IMAGES_SUBFOLDER
    if subfolder_path.exists() and subfolder_path.is_dir():
        for item in subfolder_path.iterdir():
            target = output_dir / item.name
            if not target.exists() or force:
                item.rename(target)
        try:
            subfolder_path.rmdir()
        except OSError:
            pass
    print(f"Sample images saved to {output_dir}")
    return output_dir


def download_notebooks(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download example Jupyter notebooks."""
    snapshot_download, _ = _ensure_huggingface_hub()
    repo_id = os.environ.get("UNREFLECTANYTHING_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO)
    if output_dir is None:
        output_dir = get_cache_dir(subdir="notebooks")
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        force_download=force,
        allow_patterns=[f"{NOTEBOOKS_SUBFOLDER}/*", f"{NOTEBOOKS_SUBFOLDER}/**/*"],
    )
    subfolder_path = output_dir / NOTEBOOKS_SUBFOLDER
    if subfolder_path.exists() and subfolder_path.is_dir():
        for item in subfolder_path.iterdir():
            target = output_dir / item.name
            if not target.exists() or force:
                if item.is_file():
                    item.rename(target)
                elif item.is_dir():
                    import shutil

                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(item), str(target))
        try:
            subfolder_path.rmdir()
        except OSError:
            pass
    print(f"Notebooks saved to {output_dir}")
    return output_dir


def download_configs(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download example/training config YAML files."""
    snapshot_download, _ = _ensure_huggingface_hub()
    repo_id = os.environ.get("UNREFLECTANYTHING_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO)
    if output_dir is None:
        output_dir = get_cache_dir(subdir="configs")
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        force_download=force,
        allow_patterns=[f"{CONFIGS_SUBFOLDER}/*", f"{CONFIGS_SUBFOLDER}/**/*"],
    )
    subfolder_path = output_dir / CONFIGS_SUBFOLDER
    if subfolder_path.exists() and subfolder_path.is_dir():
        for item in subfolder_path.iterdir():
            target = output_dir / item.name
            if not target.exists() or force:
                if item.is_file():
                    item.rename(target)
                elif item.is_dir():
                    import shutil

                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(item), str(target))
        try:
            subfolder_path.rmdir()
        except OSError:
            pass
    print(f"Configs saved to {output_dir}")
    return output_dir


def download_all(
    output_dir: Optional[Path] = None,
    variant: str = "default",
    force: bool = False,
) -> Path:
    """Download all assets: weights, sample images, notebooks, and configs."""
    if output_dir is None:
        output_dir = get_cache_dir()
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    download_weights(output_dir=output_dir / "weights", variant=variant, force=force)
    download_images(output_dir=output_dir / "images", force=force)
    download_notebooks(output_dir=output_dir / "notebooks", force=force)
    download_configs(output_dir=output_dir / "configs", force=force)
    print(f"All assets saved to {output_dir}")
    return output_dir


def _collect_image_paths(
    root: Path,
    extensions: Sequence[str],
) -> List[Path]:
    """Collect image paths under root matching extensions (case-insensitive)."""
    lower_exts = tuple(ext.lower() for ext in extensions)
    paths = [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in lower_exts
    ]
    return sorted(paths)


def _resolve_device(device: str) -> str:
    """Resolve device string for inference: use CUDA when available, else CPU.

    When device is 'cuda' and exactly one GPU is available, returns 'cuda'.
    When device is 'cuda' and multiple GPUs exist, returns 'cuda:0'.
    Otherwise returns the given device string (e.g. 'cuda:1', 'cpu').
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    if device == "cuda":
        return "cuda:0" if torch.cuda.device_count() > 1 else "cuda"
    return device


def _apply_config_to_options(options: Any, config: Any) -> Any:
    """Apply config overrides to inference options.

    config can be a path to YAML, or a dict with keys batch_size, device,
    brightness_threshold, resize_output, num_workers.
    """
    import yaml

    if isinstance(config, (str, Path)):
        config_path = Path(config).expanduser().resolve()
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = {}
    elif isinstance(config, dict):
        config_dict = config
    else:
        return options
    if "batch_size" in config_dict:
        options.batch_size = int(config_dict["batch_size"])
    if "device" in config_dict:
        options.device = config_dict["device"]
    if "brightness_threshold" in config_dict:
        options.brightness_threshold = float(config_dict["brightness_threshold"])
    if "resize_output" in config_dict:
        options.resize_output = bool(config_dict["resize_output"])
    if "num_workers" in config_dict:
        options.num_workers = int(config_dict["num_workers"])
    return options
