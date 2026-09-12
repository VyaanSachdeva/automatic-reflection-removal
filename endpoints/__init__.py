"""UnReflectAnything: deep learning method for removing specular reflections from RGB images.

Public API (recommended):
    inference: Run inference on images to remove reflections.
    train: Train the model.
    test: Test/evaluate a trained model.
    download: Download weights, images, or notebooks.
    evaluate: Compute evaluation metrics.
    verify: Verify dataset structure or weights integrity.
    verify_dataset: Verify dataset structure (alias for verify(what="dataset", ...)).
    cite: Get citation in various formats.
    cache: Return the cache directory (call with no arg for base, or asset name e.g. "weights").
    cache_clear: Delete cached assets (all or a specific subdir).

Legacy API (still supported):
    run_pipeline: Train or test the network (main entry for training/testing).
    run_inference: Run inference on an image directory from InferenceOptions.
    InferenceOptions: Dataclass for inference configuration.
    compute_highlight_mask: Compute binary highlight masks from RGB batch.
    get_cache_dir: Base cache directory for downloaded assets.
    ImageDirDataset: Dataset that reads images from a directory (torch Dataset).
    DEFAULT_IMAGE_EXTENSIONS: Default image file extensions for ImageDirDataset.

Example:
    >>> from unreflectanything import inference, evaluate, cache
    >>> inference("input.png", output="output.png")
    >>> results = evaluate("output.png", "reference.png")
    >>> cache("weights")
    PosixPath('/home/user/.cache/unreflectanything/weights')
"""

from __future__ import annotations

__all__ = [
    # New API (recommended)
    "inference",
    "model",
    "UnReflectModel",
    "train",
    "test",
    "download",
    "evaluate",
    "verify_dataset",
    "cite",
    "cache",
    "cache_dir",
    "cache_clear",
    # Legacy API (still supported)
    "run_pipeline",
    "run_inference",
    "InferenceOptions",
    "compute_highlight_mask",
    "get_cache_dir",
    "ImageDirDataset",
    "DEFAULT_IMAGE_EXTENSIONS",
]


def __getattr__(name: str):
    """Lazy imports so that `import unreflectanything` and CLI --help stay fast."""
    # New API functions (one module per subcommand)
    if name == "inference":
        from .inference_ import inference

        return inference
    if name == "model":
        from .model_ import model

        return model
    if name == "UnReflectModel":
        from .model_ import UnReflectModel

        return UnReflectModel
    if name == "train":
        from .train_ import train

        return train
    if name == "test":
        from .test_ import test

        return test
    if name == "download":
        from .download_ import download

        return download
    if name == "evaluate":
        from .evaluate_ import evaluate

        return evaluate
    if name == "verify":
        from .verify_ import verify

        return verify
    if name == "verify_dataset":
        from .verify_ import verify_dataset

        return verify_dataset
    if name == "cite":
        from .cite_ import cite

        return cite
    if name == "cache":
        from .cache_ import cache

        return cache
    if name == "cache_dir":
        from .cache_ import cache_dir

        return cache_dir
    if name == "cache_clear":
        from .cache_ import cache_clear

        return cache_clear

    # Legacy API
    if name == "run_pipeline":
        import main as _main

        return _main.run_pipeline
    if name == "InferenceOptions":
        from inference import InferenceOptions

        return InferenceOptions
    if name == "compute_highlight_mask":
        from inference import compute_highlight_mask

        return compute_highlight_mask
    if name == "run_inference":
        from inference import run_inference

        return run_inference
    if name == "get_cache_dir":
        from ._shared import get_cache_dir

        return get_cache_dir
    if name == "ImageDirDataset":
        from .dataset_ import ImageDirDataset

        return ImageDirDataset
    if name == "DEFAULT_IMAGE_EXTENSIONS":
        from ._shared import DEFAULT_IMAGE_EXTENSIONS

        return DEFAULT_IMAGE_EXTENSIONS

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
