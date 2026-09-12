"""Utility script for running UnReflectAnything inference on arbitrary image folders.

This module provides a command line entry point that loads
`UnReflect_Model_TokenInpainter` checkpoints and produces diffuse outputs for
every image inside a user-specified directory tree while preserving the input
directory structure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from utilities.inference import _console
from utilities.inference import UsageMonitor
from utilities.inference import (
    save_diffuse_batch,
    compute_highlight_mask,
    list_image_paths,
)
from utilities.model import load_pretrained


@dataclass
class InferenceOptions:
    """Container for CLI/YAML driven inference options.

    Attributes:
        weights_path: Absolute path to the checkpoint with the serialized model
            state. The checkpoint must contain a ``model_state_dict`` entry.
        input_dir: Root directory that holds the RGB images to be processed. The
            script traverses the directory recursively and keeps the relative
            folder layout when exporting the diffuse predictions.
        output_dir: Destination directory that will mirror the input structure.
        run: Optional run identifier used to recover metadata from an existing
            experiment directory. This is the same identifier accepted in test
            mode by ``run_pipeline``.
        runs_dir: Optional base directory that stores experiment outputs. When
            provided together with ``run``, the script loads the saved config
            artifacts for model reconstruction.
        model_config_path: Optional configuration YAML that follows the training
            template (``config/train.yaml``). When specified, the configuration
            is parsed and used to build the architecture prior to loading the
            checkpoint weights.
        batch_size: Number of images processed per forward pass. The default is
            ``4``.
        device: Preferred device string (e.g. ``"cuda"`` or ``"cpu"``).
        image_extensions: Sequence of file suffixes that are treated as valid
            images. All comparisons are case-insensitive.
        resize_output: If ``True``, resize output images to match the original
            input image dimensions. Defaults to ``True``.
        brightness_threshold: Threshold value for computing highlight masks via
            intensity thresholding. Pixels with brightness (average of R, G, B)
            above this value are considered highlights. Defaults to ``0.7``.
        monitor_usage: If ``True``, track and report FLOPS and energy consumption
            metrics after inference completes. Defaults to ``False``.
        num_workers: Number of parallel workers for loading images. Defaults to ``4``.
            Set to ``0`` to disable parallel loading (uses main process only).
    """

    weights_path: Path
    input_dir: Path
    output_dir: Path
    inpaint_mask_dilation: int = 11
    run: Optional[str] = None
    runs_dir: Optional[Path] = None
    model_config_path: Optional[Path] = None
    model_module: Optional[str] = None
    batch_size: int = 4
    device: str = "cuda"
    image_extensions: Sequence[str] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )
    resize_output: bool = True
    brightness_threshold: float = 0.7
    monitor_usage: bool = False
    num_workers: int = 4
    composite: bool = False


def run_inference(options: InferenceOptions) -> None:
    """Execute end-to-end inference on the dataset described by ``options``."""
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import functional as TF
    from rich.progress import Progress, SpinnerColumn, TextColumn

    desired_device = torch.device(
        options.device if torch.cuda.is_available() else "cpu"
    )
    model = load_pretrained(
        weights_path=options.weights_path,
        config_path=options.model_config_path,
        device=str(desired_device),
        verbose=False,
        model_module=options.model_module,
    )
    target_side = model.dinov3.config["image_size"]
    target_size = (target_side, target_side)

    image_paths = list_image_paths(options.input_dir, options.image_extensions)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Initialize usage monitor if requested
    monitor = None
    if options.monitor_usage:
        monitor = UsageMonitor(desired_device, model)
        monitor.start_monitoring()

    print(
        f"Starting inference over '{len(image_paths)}' images with batch size {options.batch_size}"
    )
    if options.num_workers > 0:
        print(f"Using '{options.num_workers}' parallel workers for image loading")

    # Dataset class defined here so torch/PIL are only loaded when run_inference runs
    class ImageDataset(Dataset):
        """Dataset for loading and preprocessing images in parallel."""

        def __init__(self, paths: List[Path], target_size: Tuple[int, int]):
            self.paths = paths
            self.target_size = target_size

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int):
            path = self.paths[idx]
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                original_size = rgb_img.size[::-1]
                tensor = TF.to_tensor(rgb_img)
                resized = TF.resize(tensor, self.target_size, antialias=True)
            size_tensor = torch.tensor(original_size, dtype=torch.int32)
            return resized, size_tensor, str(path)

    dataset = ImageDataset(image_paths, target_size)
    dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        pin_memory=desired_device.type == "cuda",
        persistent_workers=options.num_workers > 0,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed}/{task.total}"),
        console=_console(),
        transient=False,
    ) as progress:
        task_id = progress.add_task("Processing", total=len(image_paths))

        for batch_idx, batch_data in enumerate(dataloader):
            # Unpack batch data: (images, original_sizes, paths)
            images, size_tensors, batch_paths = batch_data

            # Convert to proper format
            # images: [B, 3, H, W] - already stacked tensors from DataLoader
            rgb_batch = images.to(device=desired_device, dtype=torch.float32)

            # size_tensors: [B, 2] where each row is [H, W]
            original_sizes = [
                (int(size[0].item()), int(size[1].item())) for size in size_tensors
            ]
            batch_paths = [Path(p) for p in batch_paths]

            # Update progress with first image name in batch
            if batch_paths:
                progress.update(
                    task_id, description=f"Processing {batch_paths[0].name}"
                )

            inpaint_mask_override = compute_highlight_mask(
                rgb_batch, threshold=options.brightness_threshold
            )
            inpaint_mask_dilation = options.inpaint_mask_dilation
            # Compute FLOPS on first batch if monitoring
            if monitor is not None and monitor.flops_per_image is None:
                monitor.compute_flops(rgb_batch, inpaint_mask_override)

            # Time the forward pass
            forward_start = time.time()
            with torch.no_grad():
                outputs = model(
                    {
                        "rgb": rgb_batch,
                        # "inpaint_mask_override": inpaint_mask_override,
                        "inpaint_mask_dilation": inpaint_mask_dilation,
                    }
                )
            forward_time = time.time() - forward_start

            # Record forward pass for monitoring
            if monitor is not None:
                monitor.record_forward(len(batch_paths), forward_time)

            diffuse = outputs.get("diffuse")
            if diffuse is None:
                raise KeyError("Model output does not contain 'diffuse'")
            if options.composite:
                m = outputs.get("highlight_mask")
                if m is not None:
                    import torch.nn.functional as F

                    m = F.avg_pool2d(m.float(), kernel_size=5, stride=1, padding=2)
                    m = m.clamp(0.0, 1.0)
                    if m.shape[-2:] != diffuse.shape[-2:]:
                        m = F.interpolate(
                            m,
                            size=diffuse.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    diffuse = diffuse * m + rgb_batch * (1.0 - m)
                    diffuse = diffuse.clamp(0.0, 1.0)
            save_diffuse_batch(
                diffuse,
                batch_paths,
                options.input_dir,
                options.output_dir,
                original_sizes=original_sizes if options.resize_output else None,
                resize_output=options.resize_output,
            )
            progress.advance(task_id, advance=len(batch_paths))

    # Stop monitoring and generate report
    if monitor is not None:
        monitor.stop_monitoring()
        print("")  # Empty line for spacing
        report_table = monitor.generate_report(len(image_paths))
        _console().print(report_table)
        print("")  # Empty line for spacing

    print(f"✨ Inference complete. Results saved to '{options.output_dir}'")


def main() -> None:
    """CLI entry point."""
    from endpoints.inference_ import parse_cli

    options = parse_cli()
    run_inference(options)


if __name__ == "__main__":
    main()
