import os
import sys
import warnings
from contextlib import contextmanager
from typing import Optional, TYPE_CHECKING
from pathlib import Path
from typing import Sequence, List, Tuple


class UsageMonitor:
    """Monitor FLOPS and energy consumption during inference."""

    if TYPE_CHECKING:
        from torch import Tensor
        import torch

    def __init__(self, device: "torch.device", model: "torch.nn.Module"):
        """Initialize the usage monitor.

        Args:
            device: The device being used for inference.
            model: The model to monitor.
        """
        self.device = device
        self.model = model
        self.is_cuda = device.type == "cuda"

        # FLOPS tracking
        self.flops_per_forward = None  # FLOPS for the batch used to compute it
        self.flops_per_image = None  # FLOPS per single image
        self.total_flops = 0
        self.forward_count = 0

        # Energy tracking
        self.energy_start = None
        self.energy_end = None
        self.energy_initialized = False

        # Time tracking
        self.total_forward_time = 0.0

        # Initialize GPU monitoring if available
        if self.is_cuda:
            try:
                import pynvml

                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.energy_initialized = True
            except Exception as e:
                print(
                    f"[yellow]Warning: Could not initialize GPU energy monitoring: {e}[/yellow]"
                )
                self.energy_initialized = False
        else:
            self.energy_initialized = False

    def get_gpu_info(self) -> Optional[dict]:
        """Get GPU hardware information."""
        if not self.is_cuda or not self.energy_initialized:
            return None

        try:
            import pynvml

            name = pynvml.nvmlDeviceGetName(self.handle).decode("utf-8")
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                "name": name,
                "total_memory_gb": memory_info.total / (1024**3),
            }
        except Exception:
            return None

    def start_monitoring(self):
        """Start energy monitoring."""
        if self.energy_initialized:
            try:
                import pynvml

                self.energy_start = (
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0
                )  # Convert mJ to J
            except Exception:
                self.energy_start = None

    def stop_monitoring(self):
        """Stop energy monitoring."""
        if self.energy_initialized:
            try:
                import pynvml

                self.energy_end = (
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0
                )  # Convert mJ to J
            except Exception:
                self.energy_end = None

    def compute_flops(
        self,
        rgb_batch: "Tensor",
        inpaint_mask_override: "Tensor",
        inpaint_mask_dilation: int = 11,
    ):
        """Compute FLOPS for a single forward pass.

        Args:
            rgb_batch: Input RGB batch tensor of shape [B,3,H,W].
            inpaint_mask_override: Patch mask tensor of shape [B,1,H,W].
            inpaint_mask_dilation: Dilation value for mask processing. Defaults to 11.
        """
        try:
            from fvcore.nn import FlopCountAnalysis
        except ImportError:
            return
        import torch

        if self.flops_per_forward is None:
            try:
                # Ensure model is in eval mode
                self.model.eval()

                # Ensure inputs are detached and don't require gradients
                # fvcore may have issues with tensors that have gradients or are part of computation graph
                rgb_detached = rgb_batch.detach().clone().requires_grad_(False)
                mask_detached = (
                    inpaint_mask_override.detach().clone().requires_grad_(False)
                )

                # Create a proper nn.Module wrapper for fvcore compatibility
                class ModelWrapper(torch.nn.Module):
                    """Wrapper module to ensure model outputs are properly handled by fvcore."""

                    def __init__(self, model, dilation_value):
                        super().__init__()
                        self.model = model
                        # Store dilation as a constant to avoid tracing issues
                        self.dilation_value = int(dilation_value)

                    def forward(self, input_dict):
                        """Forward pass that extracts diffuse output."""
                        # Ensure inpaint_mask_dilation is a Python int (not traced)
                        # This prevents fvcore from creating symbolic values that break max_pool2d
                        input_dict = dict(input_dict)  # Create a copy
                        input_dict["inpaint_mask_dilation"] = self.dilation_value

                        with torch.no_grad():
                            output = self.model(input_dict)
                            # Extract diffuse output and ensure it's a tensor (not dict)
                            if isinstance(output, dict):
                                diffuse = output.get("diffuse")
                                if diffuse is not None:
                                    # Ensure output is detached and doesn't require grad
                                    return diffuse.detach()
                            elif isinstance(output, torch.Tensor):
                                return output.detach()
                            return output

                # Use the provided dilation value (ensured to be int)
                dilation_value = int(inpaint_mask_dilation)

                # Create wrapper module with dilation as a constant attribute
                model_wrapper = ModelWrapper(self.model, dilation_value)
                model_wrapper.eval()

                # Create a dummy input dict matching the model's expected input
                # Note: dilation_value is stored in the wrapper, not passed in dict
                # to avoid fvcore tracing issues
                dummy_input = {
                    "rgb": rgb_detached,
                    "inpaint_mask_override": mask_detached,
                    # inpaint_mask_dilation will be set by the wrapper's forward method
                }
                # Suppress fvcore warnings about unsupported operators
                # Temporarily patch round() to handle tensors for fvcore compatibility
                import builtins

                original_round = builtins.round

                def tensor_aware_round(value, ndigits=None):
                    """Round function that handles both scalars and tensors."""
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            return original_round(float(value.item()), ndigits)
                        else:
                            # For multi-element tensors, return as-is or round element-wise
                            return value
                    return original_round(value, ndigits)

                with suppress_stdout_stderr():
                    # Temporarily replace round() to handle tensors
                    builtins.round = tensor_aware_round
                    try:
                        # Use FlopCountAnalysis API with wrapper function
                        flop_counter = FlopCountAnalysis(model_wrapper, dummy_input)
                        # Get total FLOPS for the batch
                        batch_flops = flop_counter.total()
                    except (TypeError, AttributeError, RuntimeError) as inner_e:
                        # If there's an error, try by_operator as fallback
                        if "round" in str(inner_e).lower() or "__round__" in str(
                            inner_e
                        ):
                            # The rounding error - try to work around it using by_operator
                            try:
                                by_op = flop_counter.by_operator()
                                # Sum all operator FLOPS, handling tensor values
                                batch_flops = 0.0
                                for op_name, flops in by_op.items():
                                    if isinstance(flops, torch.Tensor):
                                        batch_flops += float(flops.item())
                                    else:
                                        batch_flops += float(flops)
                            except Exception:
                                # If that also fails, re-raise the original error
                                raise inner_e
                        else:
                            raise inner_e
                    finally:
                        # Restore original round function
                        builtins.round = original_round

                    # Convert to Python scalar if it's a Tensor
                    if isinstance(batch_flops, torch.Tensor):
                        batch_flops = float(batch_flops.item())
                    else:
                        batch_flops = float(batch_flops)
                self.flops_per_forward = batch_flops
                # Compute FLOPS per image by dividing by batch size
                batch_size = rgb_batch.shape[0]
                self.flops_per_image = batch_flops / batch_size
            except Exception as e:
                print(f"[yellow]Warning: Could not compute FLOPS: {e}[/yellow]")
                self.flops_per_forward = None
                self.flops_per_image = None

    def record_forward(self, batch_size: int, forward_time: float):
        """Record a forward pass.

        Args:
            batch_size: Number of images in the batch.
            forward_time: Time taken for the forward pass in seconds.
        """
        self.forward_count += batch_size
        self.total_forward_time += forward_time

        if self.flops_per_image is not None:
            # Add FLOPS for this batch: FLOPS per image * batch size
            self.total_flops += self.flops_per_image * batch_size

    def get_energy_consumption_wh(self) -> Optional[float]:
        """Get total energy consumption in Watt-hours."""
        if self.energy_start is None or self.energy_end is None:
            return None
        # Energy is in Joules, convert to Wh: 1 Wh = 3600 J
        energy_joules = self.energy_end - self.energy_start
        return energy_joules / 3600.0

    def generate_report(self, total_images: int):
        """Generate a tabular report of usage metrics.

        Args:
            total_images: Total number of images processed.

        Returns:
            A rich Table with the usage report.
        """
        from rich.table import Table

        table = Table(
            title="Energy and Compute Usage Report",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")

        # Hardware info
        if self.is_cuda:
            gpu_info = self.get_gpu_info()
            if gpu_info:
                table.add_row("Hardware", gpu_info["name"], "CUDA GPU")
                table.add_row("GPU Memory", f"{gpu_info['total_memory_gb']:.2f}", "GB")
            else:
                table.add_row("Hardware", "NVIDIA GPU", "CUDA")
        else:
            table.add_row("Hardware", "CPU", "CPU")

        table.add_row("", "", "")  # Separator

        # FLOPS metrics
        if self.flops_per_image is not None:
            # Per forward pass (for the batch size used during computation)
            if self.flops_per_forward is not None:
                flops_forward_g = self.flops_per_forward / 1e9
                table.add_row(
                    "FLOPS (Forward Pass)", f"{flops_forward_g:.2f}", "GFLOPs"
                )

            # Per image
            flops_per_image_g = self.flops_per_image / 1e9
            table.add_row("FLOPS (Per Image)", f"{flops_per_image_g:.2f}", "GFLOPs")

            # Total dataset
            total_flops_t = self.total_flops / 1e12
            table.add_row("FLOPS (Total Dataset)", f"{total_flops_t:.4f}", "TFLOPs")
        else:
            table.add_row("FLOPS (Forward Pass)", "N/A", "(fvcore not available)")
            table.add_row("FLOPS (Per Image)", "N/A", "")
            table.add_row("FLOPS (Total Dataset)", "N/A", "")

        table.add_row("", "", "")  # Separator

        # Energy metrics
        energy_wh = self.get_energy_consumption_wh()
        if energy_wh is not None:
            # Per forward pass (average)
            if self.forward_count > 0:
                energy_per_image = energy_wh / self.forward_count
                table.add_row("Energy (Per Image)", f"{energy_per_image:.6f}", "Wh")

            # Total dataset
            table.add_row("Energy (Total Dataset)", f"{energy_wh:.4f}", "Wh")

            # Additional environmental metrics
            # CO2 equivalent (assuming average grid mix: ~0.5 kg CO2/kWh)
            co2_kg = energy_wh * 0.0005  # Convert Wh to kWh then multiply by 0.5
            table.add_row("CO2 Equivalent (Est.)", f"{co2_kg:.6f}", "kg CO2")
        else:
            table.add_row("Energy (Per Image)", "N/A", "(pynvml not available)")
            table.add_row("Energy (Total Dataset)", "N/A", "")
            table.add_row("CO2 Equivalent (Est.)", "N/A", "")

        table.add_row("", "", "")  # Separator

        # Performance metrics
        if self.total_forward_time > 0:
            avg_time_per_image = (
                self.total_forward_time / self.forward_count
                if self.forward_count > 0
                else 0
            )
            table.add_row(
                "Avg Time (Per Image)", f"{avg_time_per_image:.4f}", "seconds"
            )
            table.add_row(
                "Total Inference Time", f"{self.total_forward_time:.2f}", "seconds"
            )

            if energy_wh is not None and self.total_forward_time > 0:
                avg_power = (
                    energy_wh * 3600
                ) / self.total_forward_time  # Convert Wh to J, then divide by time
                table.add_row("Average Power", f"{avg_power:.2f}", "Watts")

        table.add_row("", "", "")  # Separator
        table.add_row("Total Images Processed", str(total_images), "images")

        return table


def _console():
    """Lazy import for rich.Console so heavy deps are not loaded when only InferenceOptions is used."""
    from rich.console import Console

    return Console()


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout, stderr, and warnings output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_warnings_showwarning = warnings.showwarning
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            # Suppress warnings
            warnings.filterwarnings("ignore")
            warnings.showwarning = lambda *args, **kwargs: None
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            warnings.showwarning = old_warnings_showwarning
            warnings.resetwarnings()


def save_diffuse_batch(
    diffuse_batch: "Tensor",
    batch_paths: Sequence[Path],
    input_root: Path,
    output_root: Path,
    original_sizes: Optional[List[Tuple[int, int]]] = None,
    resize_output: bool = True,
) -> None:
    """Write diffuse predictions to disk preserving directory structure.

    Args:
        diffuse_batch: Tensor of shape ``[B,3,H,W]`` containing diffuse predictions.
        batch_paths: Sequence of input image paths.
        input_root: Root directory of input images.
        output_root: Root directory for output images.
        original_sizes: Optional list of original image sizes ``[(H, W), ...]``.
            Required if ``resize_output`` is ``True``.
        resize_output: If ``True``, resize output images to original dimensions.
    """
    from torchvision.transforms import functional as TF

    diffuse_batch = diffuse_batch.clamp_(0.0, 1.0).cpu()
    for idx, (tensor, input_path) in enumerate(zip(diffuse_batch, batch_paths)):
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if resize_output and original_sizes is not None:
            original_size = original_sizes[idx]
            # TF.resize expects (H, W) but PIL Image.size is (W, H), so we reverse
            tensor = TF.resize(tensor, original_size, antialias=True)

        image = TF.to_pil_image(tensor)
        image.save(output_path)


def run_model(
    model: "torch.nn.Module",
    rgb_batch: "Tensor",
    inpaint_mask_dilation: int = 11,
) -> "Tensor":
    """Run minimal-overhead model forward pass on a batch of RGB images.

    This is the core inference function optimized for speed. It performs only
    the essential model forward pass without any file I/O, monitoring, or
    progress reporting.

    Args:
        model: The loaded UnReflectAnything model in eval mode.
        rgb_batch: Input RGB tensor of shape [B, 3, H, W] with values in [0, 1].
            The spatial dimensions should match the model's expected input size
            (typically 448x448).
        inpaint_mask_dilation: Dilation kernel size for the inpainting mask.
            Defaults to 11.

    Returns:
        Diffuse output tensor of shape [B, 3, H, W] with values in [0, 1].

    Example:
        >>> model = load_model(options, device)
        >>> rgb = torch.rand(4, 3, 448, 448, device='cuda')  # [B, C, H, W]
        >>> diffuse = run_model(model, rgb)  # [B, 3, H, W]
    """
    import torch

    model.eval()
    with torch.no_grad():
        outputs = model(
            {
                "rgb": rgb_batch,
                "inpaint_mask_dilation": inpaint_mask_dilation,
            }
        )

    diffuse = outputs.get("diffuse")
    if diffuse is None:
        raise KeyError("Model output does not contain 'diffuse'")

    return diffuse.clamp(0.0, 1.0)


def list_image_paths(root: Path, extensions: Sequence[str]) -> List[Path]:
    """Collect image files under ``root`` matching the provided extensions."""

    lower_exts = tuple(ext.lower() for ext in extensions)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in lower_exts
    ]
    if not files:
        raise RuntimeError(f"No images found under {root}")
    sorted_files = sorted(files)
    print(f"Discovered '{len(sorted_files)}' images under {root}")
    return sorted_files


def load_image_batch(
    paths: Sequence[Path], target_size: Tuple[int, int], device: "torch.device"
):
    """Load a batch of images into a tensor of shape ``[B,3,H,W]``.

    This function maintains backward compatibility but uses sequential loading.
    For better performance, use the DataLoader-based approach in run_inference.

    Returns:
        A tuple containing:
        - The batch tensor of shape ``[B,3,H,W]``
        - A list of original image sizes ``[(H, W), ...]`` for each image
    """
    import torch
    from PIL import Image
    from torchvision.transforms import functional as TF

    images = []
    original_sizes = []
    for path in paths:
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            original_sizes.append(
                rgb_img.size[::-1]
            )  # PIL size is (W, H), we need (H, W)
            tensor = TF.to_tensor(rgb_img)
            resized = TF.resize(tensor, target_size, antialias=True)
            images.append(resized)
    batch = torch.stack(images, dim=0)
    return batch.to(device=device, dtype=torch.float32), original_sizes


def compute_highlight_mask(rgb_batch: "Tensor", threshold: float = 0.7) -> "Tensor":
    """Compute binary highlight masks via intensity thresholding.

    The mask uses the simple brightness (average of R, G, B channels)
    and sets locations with brightness > threshold to one. The tensor shape is ``[B,1,H,W]``.

    Args:
        rgb_batch: Input RGB batch tensor of shape ``[B,3,H,W]``.
        threshold: Brightness threshold value. Pixels with brightness above this
            value are considered highlights. Defaults to ``0.7``.

    Returns:
        Binary mask tensor of shape ``[B,1,H,W]``.
    """
    # Compute brightness as the mean across the channel dimension (R,G,B)
    brightness = (
        rgb_batch.mean(dim=1, keepdim=True) / rgb_batch.mean(dim=1, keepdim=True).max()
    )  # [B,1,H,W]
    mask = (brightness > threshold).to(rgb_batch.dtype)
    return mask
