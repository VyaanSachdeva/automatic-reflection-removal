"""Verification API (dataset structure and weights) for UnReflectAnything."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

from os import PathLike

__all__ = ["verify", "verify_dataset"]


def _verify_weights_impl(
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    model_config_path: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Verify weights file exists and loads into model with no key alignment errors."""
    import torch

    from ._shared import DEFAULT_WEIGHTS_FILENAME, get_cache_dir
    from .model_ import model

    if weights_path is None:
        resolved = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
    else:
        resolved = Path(weights_path).expanduser().resolve()

    if not resolved.exists():
        print(f"Weights file not found: {resolved}")
        return False
    print(
        f"Found weights file: {resolved}\nLoading weights and verifying key alignments..."
    )

    config_path = Path(model_config_path).expanduser().resolve() if model_config_path else None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model(
            pretrained=True,
            weights_path=resolved,
            config_path=config_path,
            device=str(device),
            strict=True,
            verbose=False,
        )
        print("[SUCCESS]  Weights verified: loaded into model with no key alignment errors.")
        return True
    except (KeyError, RuntimeError, FileNotFoundError) as e:
        print(f"[FAILED]  Weights verification failed: {e}")
        print("Download the model weights with 'unreflectanything download --weights'")
        return False


def _verify_dataset_impl(
    dataset_path: Path,
    dataset_type: Optional[str] = None,
    config: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Internal implementation of dataset verification."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    from dataset import UnReflectAnything_Dataset
    from dataset.wrappers import DATASET_DEFAULTS, _identity_kwargs_for

    def _try_ds(name: str) -> Optional[UnReflectAnything_Dataset]:
        """Attempt to instantiate UnReflectAnything_Dataset with a given identity default."""
        try:
            kwargs = _identity_kwargs_for(name)
            kwargs.update({
                "root_dir": str(dataset_path),
                "target_size": (224, 224),
                "few_images": True,
            })
            ds = UnReflectAnything_Dataset(**kwargs)
            if len(ds) > 0:
                # Test load first sample to ensure it's not just an empty directory structure
                _ = ds[0]
                return ds
        except Exception:
            pass
        return None

    # 1. Explicit type provided: verify that specific wrapper/config
    if dataset_type is not None:
        ds = _try_ds(dataset_type.upper())
        if ds:
            print(f"[SUCCESS]  Dataset '{dataset_type}' verified: {len(ds)} samples found.")
            return True
        else:
            print(f"[FAILED]  Dataset verification failed for explicit type '{dataset_type}' at {dataset_path}")
            return False

    # 2. Auto-detection (wrapper-agnostic discovery)
    print(f"Auto-detecting dataset structure at: {dataset_path}")

    # First, try generic defaults (works for most standardized datasets)
    # Using 'RGBP' as a placeholder to get _GENERIC defaults from wrappers.py
    ds = _try_ds("RGBP")
    if ds:
        print(f"[SUCCESS]  Detected standard dataset structure: {len(ds)} samples found.")
        return True

    # Second, try all known configurations in DATASET_DEFAULTS
    # This covers datasets with non-standard folder names or extensions
    for name in DATASET_DEFAULTS.keys():
        ds = _try_ds(name)
        if ds:
            print(f"[SUCCESS]  Detected structure matching type '{name}': {len(ds)} samples found.")
            return True

    # Third, structural fallback: scan subdirectories for common patterns
    common_rgb_dirs = ["rgb", "frames", "video_frames", "specular", "diffuse", "raw"]
    common_exts = [".png", ".jpg", ".jpeg", ".npy"]

    try:
        scenes = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        for scene in scenes[:5]:  # Check first few potential scene directories
            for rgb_dir in common_rgb_dirs:
                if (scene / rgb_dir).is_dir():
                    for ext in common_exts:
                        try:
                            ds = UnReflectAnything_Dataset(
                                root_dir=str(dataset_path),
                                rgb_dir_name=rgb_dir,
                                rgb_ext=ext,
                                few_images=True,
                                target_size=(224, 224),
                            )
                            if len(ds) > 0:
                                _ = ds[0]
                                print(f"[SUCCESS]  Detected custom dataset structure: rgb_dir='{rgb_dir}', ext='{ext}' ({len(ds)} samples)")
                                return True
                        except Exception:
                            continue
    except Exception:
        pass

    print(f"[FAILED]  Dataset verification failed: Could not find a valid structure in {dataset_path}")
    print("Ensure the directory contains subfolders (scenes), each containing an 'rgb' (or similar) folder with images.")
    return False


def verify(
    what: Literal["dataset", "weights"],
    path: Optional[Union[str, PathLike, Path]] = None,
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    dataset_type: Optional[str] = None,
    config: Optional[Union[str, PathLike, Path]] = None,
    model_config_path: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Verify either dataset structure or weights integrity.

    - **dataset**: Checks that the directory at `path` has the expected
      structure for the given dataset type (or auto-detects). Requires `path`.
    - **weights**: Checks that the weights file exists and loads into the
      model with no state_dict key alignment errors.

    Args:
        what: Either "dataset" or "weights".
        path: Dataset root directory (required when what="dataset").
        weights_path: Path to weights file (optional when what="weights").
        dataset_type: Dataset type for dataset verification.
        config: Optional config for dataset verification.
        model_config_path: Optional model config YAML for weights verification.

    Returns:
        True if verification passed, False otherwise.
    """
    if what == "dataset":
        if path is None:
            raise ValueError("path is required when what='dataset'")
        return _verify_dataset_impl(
            dataset_path=Path(path).expanduser().resolve(),
            dataset_type=dataset_type,
            config=config,
        )
    elif what == "weights":
        return _verify_weights_impl(
            weights_path=weights_path,
            model_config_path=model_config_path,
        )
    else:
        raise ValueError(f"what must be 'dataset' or 'weights', got {what!r}")


def verify_dataset(
    path: Union[str, PathLike, Path],
    dataset_type: Optional[str] = None,
    config: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Verify that a dataset has the correct structure for training/testing.

    Convenience wrapper around verify(what="dataset", path=path, ...).
    """
    return verify(
        what="dataset",
        path=path,
        dataset_type=dataset_type,
        config=config,
    )
