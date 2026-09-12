import torch
import torch.nn as nn
import os
from typing import Optional
import torch.nn.functional as F
from pathlib import Path
from typing import Union, Any


class DataParallelWrapper(nn.Module):
    """
    Adapter so a model that expects forward(model_input_dict) can be wrapped with
    nn.DataParallel, which scatters tensor arguments (not dicts). The Engine
    passes (rgb, inpaint_mask_override, inpaint_mask_dilation); this builds the
    dict and calls the inner module.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(
        self,
        rgb,
        inpaint_mask_override=None,
        inpaint_mask_dilation=None,
        just_extract_tokens=False,
    ):
        model_input_dict = {
            "rgb": rgb,
            "inpaint_mask_override": inpaint_mask_override,
            "inpaint_mask_dilation": inpaint_mask_dilation,
        }
        return self.module(model_input_dict, just_extract_tokens=just_extract_tokens)

    def __getattr__(self, name):
        if name == "module":
            try:
                return self._modules["module"]
            except KeyError:
                raise AttributeError(
                    "'DataParallelWrapper' object has no attribute 'module'"
                ) from None
        return getattr(self.module, name)


def _is_instance_or_cfg(x, cls):
    """Return 'instance' if x is an instance of cls, 'cfg' if dict, else raise."""
    if isinstance(x, cls):
        return "instance"
    if isinstance(x, dict):
        return "cfg"
    raise TypeError(f"Expected {cls.__name__} instance or dict config, got {type(x)}.")


def _build(component, cls):
    """
    Build a component given either an instance of `cls` or a config dict
    with kwargs for cls(**config_dict).
    """
    kind = _is_instance_or_cfg(component, cls)
    if kind == "instance":
        return component
    return cls(component)  # Pass dict as single argument for config-based constructors


def get_model_parameter_summary(model):
    """
    Generate a comprehensive parameter summary for  UnReflect_Model models.

    Args:
        model: UnReflect_Model instance

    Returns:
        dict: Detailed parameter summary with counts and breakdowns
    """
    allowed_types = ("UnReflect_Model")
    if model.__class__.__name__ not in allowed_types:
        raise ValueError("Model must UnReflect_Model")

    def count_parameters(module, trainable_only=False):
        """Count parameters in a module."""
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in module.parameters())

    def count_parameters_by_name(module, name_patterns):
        """Count parameters for modules matching name patterns."""
        total_params = 0
        trainable_params = 0

        for name, child in module.named_modules():
            if any(pattern in name for pattern in name_patterns):
                total_params += sum(p.numel() for p in child.parameters())
                trainable_params += sum(
                    p.numel() for p in child.parameters() if p.requires_grad
                )

        return total_params, trainable_params

    # Initialize summary
    summary = {
        "model_type": model.__class__.__name__,
        "total_parameters": 0,
        "trainable_parameters": 0,
        "frozen_parameters": 0,
        "components": {},
    }

    # RGB Encoder (DINOv3)
    dinov3_total, dinov3_trainable = (
        count_parameters(model.dinov3, trainable_only=False),
        count_parameters(model.dinov3, trainable_only=True),
    )
    summary["components"]["rgb_encoder"] = {
        "total": dinov3_total,
        "trainable": dinov3_trainable,
        "frozen": dinov3_total - dinov3_trainable,
        "description": "DINOv3 backbone for RGB feature extraction",
    }

    # POL components (only for RGBPOLDecomposer)
    if model.__class__.__name__ == "RGBPOLDecomposer":
        # POL Preprocessing
        pol_pre_total, pol_pre_trainable = (
            count_parameters(model.pol_pre, trainable_only=False),
            count_parameters(model.pol_pre, trainable_only=True),
        )
        summary["components"]["pol_preprocessing"] = {
            "total": pol_pre_total,
            "trainable": pol_pre_trainable,
            "frozen": pol_pre_total - pol_pre_trainable,
            "description": "Polarization preprocessing (AoLP, DoLP → [cos2θ, sin2θ, DoLP])",
        }

        # POL Encoder
        pol_enc_total, pol_enc_trainable = (
            count_parameters(model.pol_enc, trainable_only=False),
            count_parameters(model.pol_enc, trainable_only=True),
        )
        summary["components"]["pol_encoder"] = {
            "total": pol_enc_total,
            "trainable": pol_enc_trainable,
            "frozen": pol_enc_total - pol_enc_trainable,
            "description": "POLViT encoder for polarization feature extraction",
        }

        # Cross-attention modules
        cross_total, cross_trainable = (
            count_parameters(model.cross, trainable_only=False),
            count_parameters(model.cross, trainable_only=True),
        )
        summary["components"]["cross_attention"] = {
            "total": cross_total,
            "trainable": cross_trainable,
            "frozen": cross_total - cross_trainable,
            "description": "RGB-POL cross-attention fusion modules",
        }

    # Decoders
    decoders = {
        "specular_decoder": model.decS,
        "diffuse_decoder": model.decD,
        "highlight_decoder": model.decH,
    }

    for name, decoder in decoders.items():
        dec_total, dec_trainable = (
            count_parameters(decoder, trainable_only=False),
            count_parameters(decoder, trainable_only=True),
        )
        summary["components"][name] = {
            "total": dec_total,
            "trainable": dec_trainable,
            "frozen": dec_total - dec_trainable,
            "description": f"DPT decoder for {name.replace('_decoder', '')} component",
        }

    # Calculate totals
    summary["total_parameters"] = sum(
        comp["total"] for comp in summary["components"].values()
    )
    summary["trainable_parameters"] = sum(
        comp["trainable"] for comp in summary["components"].values()
    )
    summary["frozen_parameters"] = (
        summary["total_parameters"] - summary["trainable_parameters"]
    )

    return summary


def print_model_parameter_summary(model, detailed=True):
    """
    Print a formatted parameter summary for RGBPOLDecomposer or UnReflect_Model models.

    Args:
        model: RGBPOLDecomposer or UnReflect_Model instance
        detailed: Whether to print detailed breakdown by component
    """
    summary = get_model_parameter_summary(model)

    print(f"\n{'=' * 60}")
    print(f"MODEL PARAMETER SUMMARY: {summary['model_type']}")
    print(f"{'=' * 60}")

    # Overall statistics
    print("\n📊 OVERALL STATISTICS:")
    print(f"   Total Parameters:     {summary['total_parameters']:,}")
    print(f"   Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"   Frozen Parameters:    {summary['frozen_parameters']:,}")
    print(
        f"   Trainable Ratio:      {summary['trainable_parameters'] / summary['total_parameters'] * 100:.1f}%"
    )

    if detailed:
        print("\n🔍 DETAILED BREAKDOWN:")
        print(
            f"{'Component':<25} {'Total':<12} {'Trainable':<12} {'Frozen':<12} {'Ratio':<8}"
        )
        print(f"{'-' * 25} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 8}")

        for comp_name, comp_data in summary["components"].items():
            ratio = (
                comp_data["trainable"] / comp_data["total"] * 100
                if comp_data["total"] > 0
                else 0
            )
            print(
                f"{comp_name:<25} {comp_data['total']:<12,} {comp_data['trainable']:<12,} {comp_data['frozen']:<12,} {ratio:<7.1f}%"
            )

        print("\n📝 COMPONENT DESCRIPTIONS:")
        for comp_name, comp_data in summary["components"].items():
            print(f"   • {comp_name}: {comp_data['description']}")

    print(f"\n{'=' * 60}")


def get_model_size_mb(model):
    """
    Calculate model size in MB (approximate).

    Args:
        model: RGBPOLDecomposer or UnReflect_Model instance

    Returns:
        float: Model size in MB
    """
    summary = get_model_parameter_summary(model)
    # Assuming float32 (4 bytes per parameter)
    size_mb = summary["total_parameters"] * 4 / (1024 * 1024)
    return size_mb


def load_best_model_by_run(
    run_identifier: str,
    device: Optional[torch.device] = None,
    runs_dir: Optional[str] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Load and return a model instantiated from a run's best checkpoint.

    Args:
        run_identifier: Run name or ID used to locate the run directory.
        device: Optional torch.device. Defaults to CUDA if available else CPU.
        runs_dir: Optional absolute path to the root runs directory. If None, uses the
                  same resolution logic as the training engine (RESUlTS_DIR env or default).
        verbose: If True, print/log progress information. Defaults to True.

    Returns:
        nn.Module: The model put on the requested device, loaded with best weights, set to eval().

    Raises:
        FileNotFoundError: If the run or best checkpoint cannot be found.
        RuntimeError: If model instantiation or state loading fails.
    """
    # Resolve device
    load_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Loading model on device: {load_device}")

    # Resolve runs directory using same logic as engine initializers if not provided
    if runs_dir is None:
        try:
            from utilities import (
                engine_initializers as initialize,
            )  # local import to avoid cycles

            runs_dir = initialize.device_and_directories({})["runs_dir"]
        except Exception:
            # Fallback: environment variable or repo default
            runs_dir = os.path.expandvars(
                os.getenv(
                    "RESULTS_DIR",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"),
                )
            )

    # Find the run and its models directory
    try:
        from utilities.run_resume import get_resume_info  # local import to avoid cycles

        resume_info = get_resume_info(run_identifier, runs_dir)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to resolve run '{run_identifier}' in runs_dir '{runs_dir}': {e}"
        )

    if resume_info is None:
        raise FileNotFoundError(f"Run not found or invalid: {run_identifier}")

    models_dir = resume_info.get("models_dir")
    run_dir = resume_info.get("run_dir")
    if verbose:
        print(f"Found run directory: {run_dir}")
    if models_dir is None or not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found for run: {run_identifier}")

    # Prefer full_model_weights.pt (EarlyStopping), fallback to best_model.pt (engine best)
    candidate_paths = [
        os.path.join(models_dir, "full_model_weights.pt"),
        os.path.join(models_dir, "best_model.pt"),
    ]
    best_ckpt = next((p for p in candidate_paths if os.path.exists(p)), None)
    if best_ckpt is None:
        raise FileNotFoundError(
            f"Best checkpoint not found. Looked for: {', '.join(candidate_paths)}"
        )
    if verbose:
        print(f"Loading checkpoint from: {best_ckpt}")

    # Load checkpoint and reconstruct model from saved config
    if not os.path.exists(best_ckpt):
        if best_ckpt.endswith(".pt"):
            alt_best_ckpt = best_ckpt[:-3] + ".pth"
        elif best_ckpt.endswith(".pth"):
            alt_best_ckpt = best_ckpt[:-4] + ".pt"
            alt_best_ckpt = None
        if alt_best_ckpt and os.path.exists(alt_best_ckpt):
            best_ckpt = alt_best_ckpt

    checkpoint = torch.load(best_ckpt, map_location=load_device, weights_only=False)

    # Prefer config packaged inside checkpoint; else use run's config from resume_info
    saved_config = checkpoint.get("config", resume_info.get("config"))
    if saved_config is None:
        # As a last resort, try to read config.json from the run directory
        import json

        config_path = os.path.join(run_dir, "config.json") if run_dir else None
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved_config = json.load(f)
        else:
            raise RuntimeError("No configuration found to rebuild model architecture")

    # Ensure DotMap for the model factory
    try:
        from dotmap import DotMap  # local import

        if not isinstance(saved_config, DotMap):
            saved_config = DotMap(saved_config)
    except Exception:
        # If DotMap import fails, attempt to use raw dict (factory expects DotMap but guard anyway)
        pass

    # Build model using the main factory to ensure identical architecture
    try:
        from utilities.config import (
            create_model_from_config,
        )  # local import to avoid top-level cycle

        model = create_model_from_config(saved_config, load_device, verbose=verbose)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate model from saved config: {e}")
    if verbose:
        print(f"Model created: {model.__class__.__name__}")

    # Load weights (handle both full checkpoint and raw state_dict formats)
    state_dict = checkpoint.get("model_state_dict", None)
    if state_dict is None:
        # If the file is a plain state_dict
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if verbose:
        if len(missing) > 0:
            print(f"Warning: Missing keys when loading checkpoint: {len(missing)} keys")
        if len(unexpected) > 0:
            print(
                f"Warning: Unexpected keys when loading checkpoint: {len(unexpected)} keys"
            )
    if len(unexpected) > 0:
        # Not fatal; but surface to caller as warning via exception message for clarity
        # Users can inspect and decide to ignore
        pass

    model = model.to(load_device).eval()
    if verbose:
        print("Model loaded successfully and set to eval mode")
    return model


def pixel_mask_to_patch_mask(
    mask_hw: torch.Tensor,
    patch_size: int,
    threshold: float = 0.0,
    invert: bool = False,
    soft: bool = False,
) -> torch.Tensor:
    """
    Convert pixel mask to patch mask.

    Args:
        mask_hw: (B, 1, H, W) in [0,1]
        patch_size: int, spatial size of each patch
        threshold: float, threshold for determining masked pixels
        invert: bool, if True, invert the mask
        soft: bool, if True, output soft values in [0,1] representing proportion of pixels above threshold

    Returns:
        patch_mask: (B, N) where N=(H/P)*(W/P)
            - If soft=False: boolean tensor (patch is masked if ANY pixel is above threshold)
            - If soft=True: float tensor in [0,1] (proportion of pixels above threshold in each patch)
    """
    if mask_hw.dtype == torch.bool:
        m = mask_hw.float()
    else:
        m = (mask_hw > threshold).float()
    if soft:
        # Use average pooling to get proportion of pixels above threshold in each patch
        m_small = F.avg_pool2d(m, kernel_size=patch_size, stride=patch_size)
        pm = m_small.flatten(1)  # (B, N) in [0,1]
    else:
        # Use max pooling to check if ANY pixel is above threshold
        m_small = F.max_pool2d(m, kernel_size=patch_size, stride=patch_size)
        pm = m_small.flatten(1).bool()  # (B, N) boolean
    if invert:
        if soft:
            pm = 1.0 - pm
        else:
            pm = torch.logical_not(pm)
    return pm


def feather_token_mask(pm_soft: torch.Tensor, radius_tokens: int = 1, smoothstep=True):
    """
    pm_soft: (B, N) in [0,1], 1 = needs inpainting
    returns: (B, N) in [0,1], softly feathered over the token grid
    """
    B, N = pm_soft.shape
    H = int(round(N**0.5))
    assert H * H == N, "N must be a perfect square (flattened token grid)."
    m = pm_soft.view(B, 1, H, H)

    k = 2 * radius_tokens + 1
    pad = radius_tokens

    # Simple, differentiable feather: box blur on both the hole and its complement,
    # then recombine to keep values near edges smooth but not overly washed.
    kernel = torch.ones(1, 1, k, k, device=m.device, dtype=m.dtype) / (k * k)

    m_blur = F.conv2d(m, kernel, padding=pad)
    inv_blur = F.conv2d(1.0 - m, kernel, padding=pad)
    # Re-normalize: high where hole dominates, low where visible dominates
    out = m_blur / (m_blur + inv_blur + 1e-6)

    if smoothstep:
        # gentle edge emphasis without harsh slopes
        out = out * out * (3.0 - 2.0 * out)

    return out.clamp_(0, 1).view(B, N)


def patch_mask_to_pixel_mask(
    patch_mask: torch.Tensor, patch_size: int, soft: bool = False
) -> torch.Tensor:
    """
    Convert patch mask to pixel mask.

    Args:
        patch_mask: (B, N) boolean or float tensor, where N = (H/P)*(W/P)
        patch_size: int, spatial size of each patch
        soft: bool, if True, use bilinear interpolation for smooth transitions;
              if False, use nearest neighbor interpolation

    Returns:
        mask_hw: (B, 1, H, W) float tensor in [0,1]
    """
    B, N = patch_mask.shape
    # Compute target H and W (assuming square arrangement, filled in row-major order)
    patch_grid_size = int(N**0.5)

    # Reshape (B, N) -> (B, 1, grid, grid)
    patch_mask_grid = patch_mask.reshape(B, 1, patch_grid_size, patch_grid_size).float()
    # Upsample to (B, 1, H, W)
    if soft:
        # Use bilinear interpolation for smooth transitions
        pixel_mask = torch.nn.functional.interpolate(
            patch_mask_grid,
            scale_factor=patch_size,
            mode="bilinear",
            align_corners=False,
        )
    else:
        # Use nearest-neighbor interpolation
        pixel_mask = torch.nn.functional.interpolate(
            patch_mask_grid, scale_factor=patch_size, mode="nearest"
        )
    return pixel_mask


def load_pretrained(
    weights_path: Path,
    config_path: Optional[Union[Path, str, dict, Any]] = None,
    device: str = "cuda",
    strict: bool = False,
    verbose: bool = False,
    model_module: Optional[str] = None,
) -> "torch.nn.Module":
    """Build the model architecture and load checkpoint weights (single entry point for inference).

    Resolves configuration in order: checkpoint, run directory (if run/runs_dir set),
    config argument (path or dict), default_config_path. Then builds the model with
    create_model_from_config, loads state_dict, and returns the model in eval mode.

    Args:
        weights_path: Path to the checkpoint file (e.g. full_model_weights.pt).
        config: Optional config source: path to YAML or dict. Used if checkpoint has no config.
        device: Device string (e.g. "cuda", "cpu").
        strict: If True, load_state_dict uses strict=True.
        verbose: If True, print progress information.
        default_config_path: Optional path to default config when no other source is available.
        run: Optional run identifier for loading config from experiment directory.
        runs_dir: Optional base directory for runs (used with run).
        model_module: Optional override for config.MODEL.MODEL_MODULE.

    Returns:
        Loaded model in eval mode (e.g. UnReflect_Model_TokenInpainter).
    """
    import torch
    from utilities.config import (
        create_model_from_config,
        load_config_from_checkpoint,
        load_config_from_path_or_dict,
    )
    from dotmap import DotMap

    weights_path = Path(weights_path).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    if verbose:
        print(f"Loading checkpoint from '{weights_path}' on device '{device}'")
    if not weights_path.exists():
        if weights_path.endswith(".pt"):
            alt_weights_path = weights_path[:-3] + ".pth"
        elif weights_path.endswith(".pth"):
            alt_weights_path = weights_path[:-4] + ".pt"
            alt_weights_path = None
        if alt_weights_path and os.path.exists(alt_weights_path):
            weights_path = alt_weights_path
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if config_path is None:
        config = load_config_from_checkpoint(checkpoint)
    elif isinstance(config_path, (DotMap, dict)):
        config = load_config_from_path_or_dict(config_path)
    else:
        path = Path(config_path).expanduser().resolve()
        if path.exists():
            config = load_config_from_path_or_dict(path)
        else:
            config = load_config_from_checkpoint(checkpoint)

    #   )
    if config_path is not None and verbose:
        print(f"Model configuration loaded from {config_path}")

    if model_module is not None:
        config.MODEL.MODEL_MODULE = model_module
    config.USE_TORCH_COMPILE = False

    torch_device = torch.device(device)
    model = create_model_from_config(config, torch_device, verbose=verbose)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint
        # raise KeyError("Checkpoint does not contain model_state_dict")

    # Handle checkpoints saved with DataParallel / wrapped module: keys may be prefixed with "module."
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    # Helper: Find longest common prefix (by dot-level) among a list of keys
    def common_dot_prefix(keys):
        split_keys = [k.split(".") for k in keys if "." in k]
        if not split_keys:
            return ""
        min_parts = min(len(parts) for parts in split_keys)
        prefix = []
        for i in range(min_parts):
            ith = [parts[i] for parts in split_keys]
            if all(x == ith[0] for x in ith):
                prefix.append(ith[0])
            else:
                break
        return ".".join(prefix) + "." if prefix else ""

    common_prefix = ""
    if ckpt_keys and not (model_keys & ckpt_keys):
        # Find a common prefix among all checkpoint keys
        common_prefix = common_dot_prefix(list(ckpt_keys))
        if common_prefix and all(k.startswith(common_prefix) for k in ckpt_keys):
            # Attempt to strip it and reload
            stripped_keys = {k[len(common_prefix):]: v for k, v in state_dict.items()}
            if model_keys & set(stripped_keys.keys()):
                state_dict = stripped_keys
                if verbose:
                    print(f"Checkpoint keys were prefixed with '{common_prefix}'; stripped prefix to match model.")
        # Fallback: Handle normal "module." case if needed
        elif all(k.startswith("module.") for k in ckpt_keys):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
            if verbose:
                print("Checkpoint keys were prefixed with 'module.'; stripped prefix to match model.")

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    def top2_levels(key):
        return ".".join(key.split(".")[:2]) if "." in key else key

    if not strict:
        if missing:
            # Show only the top-2 levels for missing keys, and count for each group
            missing_top2 = [top2_levels(k) for k in missing]
            from collections import Counter
            missing_counter = Counter(missing_top2)
            print(f"Warning: missing keys when loading checkpoint (grouped by top-2 levels):")
            for k, v in missing_counter.items():
                print(f"  {k}: {v} key(s) missing")
            print(f"Total unique top-2-levels for missing keys: {len(missing_counter)}")
        if unexpected:
            # Show only the top-2 levels for unexpected keys, and count for each group
            unexpected_top2 = [top2_levels(k) for k in unexpected]
            from collections import Counter
            unexpected_counter = Counter(unexpected_top2)
            print(f"Warning: unexpected keys when loading checkpoint (grouped by top-2 levels):")
            for k, v in unexpected_counter.items():
                print(f"  {k}: {v} key(s) unexpected")
            print(f"Total unique top-2-levels for unexpected keys: {len(unexpected_counter)}")

    model.eval()
    if verbose:
        print("[SUCCESS]  Model loaded and ready for inference")
    return model
