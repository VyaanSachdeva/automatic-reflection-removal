"""Minimal config loading, model construction, and inference I/O for inference-only use.

This module does not import utilities, logger, or engine code. It is used by
endpoints/model_.py and endpoints/inference_.py so that instantiating the model
and running inference only load torch, transformers, and the models package.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import yaml
from dotmap import DotMap


def load_config_minimal(
    config: Optional[Union[dict, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> DotMap:
    """Load config from in-memory dict/DotMap or from a YAML path.

    Expects the same YAML shape as training configs: top-level "parameters"
    with keys whose values are dicts with "value". No logger, no utilities.
    """
    if config is not None:
        if isinstance(config, DotMap):
            out = config
        elif isinstance(config, dict):
            out = DotMap(config)
        else:
            out = DotMap(getattr(config, "toDict", lambda: dict(config))())
        out.USE_TORCH_COMPILE = False
        return out

    path = Path(config_path).expanduser().resolve()
    with open(path, "r") as f:
        config_yaml = yaml.safe_load(f)
    parameters = config_yaml.get("parameters", config_yaml)
    config_dict = {
        k: v.get("value") if isinstance(v, dict) else v
        for k, v in parameters.items()
        if v is not None
    }
    out = DotMap(config_dict)
    out.USE_TORCH_COMPILE = False
    return out


def create_model_from_config_minimal(
    config: DotMap,
    device: torch.device,
    verbose: bool = False,
) -> torch.nn.Module:
    """Build the UnReflect model from config without importing utilities or logger.

    Mirrors the model-building logic of utilities.config.create_model_from_config
    but only imports: torch, yaml, dotmap, importlib, and the configured models module.
    """
    model_config = config.get("MODEL", {})

    model_module_name = model_config.get("MODEL_MODULE", "models")
    models_module = importlib.import_module(model_module_name)

    target_size = None
    datasets_cfg = config.get("DATASETS", {})
    if isinstance(datasets_cfg, dict):
        for dataset_config in datasets_cfg.values():
            if isinstance(dataset_config, dict) and "TARGET_SIZE" in dataset_config:
                target_size = dataset_config["TARGET_SIZE"]
                break
    if target_size is None:
        target_size = (224, 224)
    if isinstance(target_size, list):
        target_size = tuple(target_size)

    rgb_encoder_config = model_config.get("RGB_ENCODER", {})
    encoder_name = rgb_encoder_config.get(
        "ENCODER", "facebook/dinov3-vits16-pretrain-lvd1689m"
    )
    is_convnext = "convnext" in encoder_name.lower()

    dinov3_cfg = {
        "model_name": encoder_name,
        "image_size": rgb_encoder_config.get("IMAGE_SIZE", min(target_size)),
        "freeze_backbone": rgb_encoder_config.get("FREEZE_BACKBONE", True),
        "return_selected_layers": rgb_encoder_config.get(
            "RETURN_SELECTED_LAYERS", [3, 6, 9, 12]
        ),
        "return_last_hidden_state": rgb_encoder_config.get(
            "RETURN_LAST_HIDDEN_STATE", False
        ),
        "return_as_feature_maps": is_convnext,
        "return_cls_token": False,
        "encoder_lr": rgb_encoder_config.get("RGB_ENCODER_LR", None),
    }

    decoders_config = model_config.get("DECODERS", None)
    if decoders_config is not None:
        decoders = {}
        for decoder_name, decoder_params in decoders_config.items():
            decoder_cfg = {
                "feature_dim": decoder_params.get("FEATURE_DIM", 384),
                "reassemble_out_channels": decoder_params.get(
                    "REASSEMBLE_OUT_CHANNELS", [12, 24, 48, 92]
                ),
                "reassemble_factors": decoder_params.get(
                    "REASSEMBLE_FACTORS", [4.0, 2.0, 1.0, 0.5]
                ),
                "readout_type": decoder_params.get("READOUT_TYPE", "ignore"),
                "use_bn": decoder_params.get("USE_BN", True),
                "dropout": decoder_params.get("DROPOUT", 0.0),
                "output_image_size": decoder_params.get(
                    "OUTPUT_IMAGE_SIZE", [min(target_size), min(target_size)]
                ),
                "output_channels": decoder_params.get("OUTPUT_CHANNELS", 3),
                "use_film": decoder_params.get("USE_FILM", False),
                "from_pretrained": decoder_params.get("FROM_PRETRAINED", ""),
                "decoder_lr": decoder_params.get("DECODER_LR", None),
            }
            decoders[decoder_name] = decoder_cfg
        decoder_kwargs = {"decoders": decoders}
    else:
        decoder_config = model_config.get("DECODER", {})
        decoder_cfg = {
            "feature_dim": decoder_config.get("FEATURE_DIM", 384),
            "reassemble_out_channels": decoder_config.get(
                "REASSEMBLE_OUT_CHANNELS", [12, 24, 48, 92]
            ),
            "reassemble_factors": decoder_config.get(
                "REASSEMBLE_FACTORS", [4.0, 2.0, 1.0, 0.5]
            ),
            "readout_type": decoder_config.get("READOUT_TYPE", "ignore"),
            "use_bn": decoder_config.get("USE_BN", True),
            "output_image_size": decoder_config.get(
                "OUTPUT_IMAGE_SIZE", [min(target_size), min(target_size)]
            ),
            "output_channels": decoder_config.get("OUTPUT_CHANNELS", 3),
        }
        decoder_kwargs = {
            "spec_decoder": decoder_cfg,
            "diffuse_decoder": decoder_cfg,
            "highlight_decoder": decoder_cfg,
        }

    if is_convnext:
        shared_dinov3 = models_module.DINOv3_ConvNext(dinov3_cfg).to(device)
    else:
        shared_dinov3 = models_module.DINOv3(dinov3_cfg).to(device)

    model_class_str = model_config.get("MODEL_CLASS", "UnReflect_Model_TokenInpainter")
    model_class = getattr(models_module, model_class_str)
    model_kwargs = {
        "dinov3": shared_dinov3,
        "verbose": verbose,
        **decoder_kwargs,
    }

    if model_class_str == "UnReflect_Model_TokenInpainter":
        token_inpainter_config = model_config.get("TOKEN_INPAINTER", {})
        token_inpainter_cfg = {
            "token_inpainter_class": token_inpainter_config.get(
                "TOKEN_INPAINTER_CLASS", "TokenInpainter"
            ),
            "token_inpainter_module": token_inpainter_config.get(
                "TOKEN_INPAINTER_MODULE", "models"
            ),
            "from_pretrained": token_inpainter_config.get("FROM_PRETRAINED", ""),
            "depth": token_inpainter_config.get("DEPTH", 4),
            "heads": token_inpainter_config.get("HEADS", 16),
            "drop": token_inpainter_config.get("DROP", 0.1),
            "use_positional_encoding": token_inpainter_config.get(
                "USE_POSITIONAL_ENCODING", True
            ),
            "use_final_norm": token_inpainter_config.get("USE_FINAL_NORM", True),
            "use_local_prior": token_inpainter_config.get("USE_LOCAL_PRIOR", True),
            "local_prior_weight": token_inpainter_config.get("LOCAL_PRIOR_WEIGHT", 0.5),
            "local_prior_kernel": token_inpainter_config.get("LOCAL_PRIOR_KERNEL", 5),
            "seed_noise_std": token_inpainter_config.get("SEED_NOISE_STD", 0.01),
        }
        model_kwargs["token_inpainter_cfg"] = token_inpainter_cfg

    model = model_class(**model_kwargs).to(device)
    if verbose:
        nparams = sum(p.numel() for p in model.parameters())
        print(f"Model {model.__class__.__name__} created with {nparams:,} parameters.")
    return model


# ---------------------------------------------------------------------------
# Minimal inference I/O (no utilities.inference import)
# ---------------------------------------------------------------------------


def list_image_paths_minimal(
    root: Path,
    extensions: Sequence[str],
    verbose: bool = False,
) -> List[Path]:
    """Collect image files under ``root`` matching the given extensions.

    Same behavior as utilities.inference.list_image_paths but without
    importing utilities. Used by endpoints/inference_.py.
    """
    lower_exts = tuple(ext.lower() for ext in extensions)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in lower_exts
    ]
    if not files:
        raise RuntimeError(f"No images found under {root}")
    sorted_files = sorted(files)
    if verbose:
        print(f"Discovered {len(sorted_files)} images under {root}")
    return sorted_files


def save_diffuse_batch_minimal(
    diffuse_batch: torch.Tensor,
    batch_paths: Sequence[Path],
    input_root: Path,
    output_root: Path,
    original_sizes: Optional[List[Tuple[int, int]]] = None,
    resize_output: bool = True,
) -> None:
    """Write diffuse predictions to disk preserving directory structure.

    Same behavior as utilities.inference.save_diffuse_batch but without
    importing utilities. diffuse_batch: [B, 3, H, W].
    """
    from torchvision.transforms import functional as TF

    diffuse_batch = diffuse_batch.clamp_(0.0, 1.0).cpu()
    for idx, (tensor, input_path) in enumerate(zip(diffuse_batch, batch_paths)):
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if resize_output and original_sizes is not None:
            original_size = original_sizes[idx]
            tensor = TF.resize(tensor, original_size, antialias=True)

        image = TF.to_pil_image(tensor)
        image.save(output_path)
