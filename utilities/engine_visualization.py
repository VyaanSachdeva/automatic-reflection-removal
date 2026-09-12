"""
Visualization helpers for the training Engine: building comparison panels,
individual images for wandb logging, and preparing decomposition dicts for logging.
"""
import gc
import math
from typing import Any, Callable, Optional

import torch

from utilities.model import patch_mask_to_pixel_mask
from utilities.visualization import panelize, rgb


def _make_black_image(size=(448, 448)):
    return torch.zeros(3, size[0], size[1])


def _prepare_img_tensor(t: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], bool]:
    """Returns (tensor_CHW, is_grayscale). tensor_CHW is [C,H,W] with C in {1,3}."""
    if not isinstance(t, torch.Tensor):
        return None, False
    x = t
    if x.dim() == 4:
        x = x[0]
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() == 3 and x.shape[0] > 4 and x.shape[-1] <= 4:
        x = x.permute(2, 0, 1)
    if x.dim() != 3:
        return None, False
    C = x.shape[0]
    if C >= 3:
        return x[:3].detach(), False
    return x[:1].detach(), True


# Type for the callback: (viz_dict, key, tensor, caption, batch_idx, phase, test_idx) -> None
AddImageFn = Callable[
    [dict, str, Any, str, int, Optional[str], Optional[int]], None
]


def prepare_decomposition_dicts_for_logging(
    gt_decomposition: dict,
    pred_decomposition: dict,
    diffuse_teacher_tokens: list,
    patch_inpaint_mask: torch.Tensor,
    patch_supervision_mask: torch.Tensor,
    pixel_supervision_mask: torch.Tensor,
    diffuse: torch.Tensor,
    image_size: int,
) -> None:
    """
    Prepare gt_decomposition and pred_decomposition for wandb image logging:
    add token_inpaint, token_sup, pixel_supervision_mask visuals and remove debug keys.
    Mutates both dicts in place.
    """
    if "tokens_teacher" not in gt_decomposition or "tokens_completed" not in pred_decomposition:
        return
    # Shapes: [B, npatches, embed_dim]
    _, npatches, embed_dim = gt_decomposition["tokens_teacher"][-1].shape
    patch_resolution = int(math.sqrt(npatches))
    resize = (image_size, image_size)
    teacher_last = diffuse_teacher_tokens[-1]
    teacher_chw = (
        teacher_last.reshape(-1, patch_resolution, patch_resolution, embed_dim)
        .permute(0, 3, 1, 2)
        .detach()[0]
    )
    _, pca = rgb(teacher_chw, as_tensor=True, return_pca=True)

    def _token_rgb(tokens_chw, pca_obj, resize_wh, blackout=False):
        return rgb(
            tokens_chw,
            pca=pca_obj,
            resize=resize_wh,
            as_tensor=True,
            blackout=blackout,
        )

    inpaint_pixel = patch_mask_to_pixel_mask(patch_inpaint_mask, patch_size=16).int()[0]
    sup_pixel = patch_mask_to_pixel_mask(patch_supervision_mask, patch_size=16).int()[0]
    sup_inpaint = sup_pixel * inpaint_pixel

    pred_last = (
        pred_decomposition["tokens_completed"][-1]
        .reshape(-1, patch_resolution, patch_resolution, embed_dim)
        .permute(0, 3, 1, 2)
        .detach()[0]
    )

    gt_decomposition["token_inpaint"] = _token_rgb(teacher_chw, pca, resize) * inpaint_pixel
    pred_decomposition["token_inpaint"] = _token_rgb(pred_last, pca, resize) * inpaint_pixel
    gt_decomposition["pixel_supervision_mask"] = rgb(
        pixel_supervision_mask.int(),
        resize=resize,
        as_tensor=True,
        colormap="gray",
    )
    pred_decomposition["pixel_supervision_mask"] = rgb(
        pixel_supervision_mask.int() * diffuse,
        resize=resize,
        as_tensor=True,
    )
    gt_decomposition["token_sup"] = _token_rgb(teacher_chw, pca, resize) * sup_inpaint
    pred_decomposition["token_sup"] = _token_rgb(pred_last, pca, resize) * sup_inpaint

    for key in ("masked_diffuse", "patch_mask_sup", "masked_tokens", "supervision_mask", "tokens_teacher", "specular"):
        gt_decomposition.pop(key, None)
    for key in ("tokens_completed", "tokens_inpainted"):
        pred_decomposition.pop(key, None)
    # pred_decomposition.pop("patch_mask", None)


def create_visualization_images(
    gt_decomposition: dict,
    pred_decomposition: dict,
    sample: Optional[dict],
    add_image_fn: AddImageFn,
    as_single_panel: bool = True,
    simplified: bool = True,
    batch_idx: int = 0,
    phase: Optional[str] = None,
    test_idx: Optional[int] = None,
    also_save_individual_images: bool = False,
    
) -> dict:
    """
    Build visualization dict (comparison panel and/or individual images).
    add_image_fn(viz_dict, key, tensor, caption, batch_idx, phase, test_idx) is
    called for each image to add (e.g. to log to wandb).
    """
    
    if as_single_panel:
        visualization_dict = {}
        if simplified:
            all_keys = ["diffuse","highlight","rgb_highlighted","token_sup"]
        else:
            all_keys = list(
                sorted(set(pred_decomposition.keys()) | set(gt_decomposition.keys()))
            )
        
        prediction_row = panelize(
            *[
                (
                    lambda _t: rgb(
                        _t[0] if _t[0] is not None else _make_black_image(),
                        as_tensor=True,
                        resize=(448, 448),
                        colormap=("gray" if _t[1] else None),
                        label={
                            "position": "top-left",
                            "height": 40,
                            "margin": 1 if comp_name not in pred_decomposition else 0,
                            "text": (
                                f"PRED {comp_name.capitalize()}"
                                if comp_name in pred_decomposition
                                else "NA"
                            ),
                        },
                    )
                )(
                    _prepare_img_tensor(
                        pred_decomposition.get(comp_name)
                    )
                )
                for comp_name in all_keys
            ],
            mode="horizontal",
        )

        gt_row = panelize(
            *[
                (
                    lambda _t: rgb(
                        _t[0] if _t[0] is not None else _make_black_image(),
                        as_tensor=True,
                        resize=(448, 448),
                        colormap=("gray" if _t[1] else None),
                        label={
                            "position": "top-left",
                            "height": 40,
                            "margin": 1 if comp_name not in gt_decomposition else 0,
                            "text": (
                                f"GT {comp_name.capitalize()}"
                                if comp_name in gt_decomposition
                                else "NA"
                            ),
                        },
                    )
                )(
                    _prepare_img_tensor(
                        gt_decomposition.get(comp_name)
                    )
                )
                for comp_name in all_keys
            ],
            mode="horizontal",
        )

        prediction_panel_loggable = panelize(
            prediction_row, gt_row, mode="vertical" if len(all_keys) > 1 else "horizontal", resize_to_match=False
        )
        add_image_fn(
            visualization_dict,
            "images/Comparison_panel",
            prediction_panel_loggable,
            "Comparison Panel",
            batch_idx,
            phase,
            test_idx,
        )

        if also_save_individual_images:
            if "recon" in pred_decomposition:
                add_image_fn(
                    visualization_dict,
                    "images/PRED_Reconstruction",
                    pred_decomposition["recon"],
                    "Reconstruction",
                    batch_idx,
                    phase,
                    test_idx,
                )
            for comp_name in ["specular", "diffuse", "AoP", "DoP"]:
                if (
                    comp_name in pred_decomposition
                    and isinstance(pred_decomposition[comp_name], torch.Tensor)
                    and pred_decomposition[comp_name].dim() == 4
                ):
                    display_name = comp_name.replace("_", " ").title()
                    add_image_fn(
                        visualization_dict,
                        f"images/PRED_{comp_name.capitalize()}",
                        pred_decomposition[comp_name],
                        f"Predicted {display_name} Component",
                        batch_idx,
                        phase,
                        test_idx,
                    )
            key_map = {
                "diffuse": ("images/GT_Diffuse", "Input RGB Image"),
                "rgb_highlighted": (
                    "images/GT_RGB_Highlighted",
                    "Input RGB Highlighted Image",
                ),
                "specular": ("images/GT_Specular", "Ground Truth Specular"),
            }
            for tensor_key in ["diffuse", "rgb_highlighted", "specular"]:
                if (
                    tensor_key in gt_decomposition
                    and gt_decomposition[tensor_key] is not None
                    and isinstance(gt_decomposition[tensor_key], torch.Tensor)
                    and gt_decomposition[tensor_key].dim() == 4
                ):
                    if tensor_key in key_map:
                        key, caption = key_map[tensor_key]
                        add_image_fn(
                            visualization_dict,
                            key,
                            gt_decomposition[tensor_key],
                            caption,
                            batch_idx,
                            phase,
                            test_idx,
                        )

        return visualization_dict

    # Non-panel branch
    visualization_dict = {}
    if "recon" in pred_decomposition:
        add_image_fn(
            visualization_dict,
            "images/PRED_Reconstruction",
            pred_decomposition["recon"],
            "Reconstruction",
            batch_idx,
            phase,
            test_idx,
        )
    for comp_name, comp_tensor in pred_decomposition.items():
        if (
            comp_name != "recon"
            and isinstance(comp_tensor, torch.Tensor)
            and comp_tensor.dim() == 4
        ):
            display_name = comp_name.replace("_", " ").title()
            add_image_fn(
                visualization_dict,
                f"images/PRED_{comp_name.capitalize()}",
                comp_tensor,
                f"Predicted {display_name} Component",
                batch_idx,
                phase,
                test_idx,
            )
    input_images = [
        ("images/GT_Diffuse", "diffuse", "Input RGB Image"),
        ("images/GT_RGB_Highlighted", "rgb_highlighted", "Input RGB Highlighted Image"),
        ("images/GT_Highlight", "highlight", "Input RGB Highlighted Image"),
    ]
    for key, tensor_key, caption in input_images:
        if (
            tensor_key in gt_decomposition
            and gt_decomposition[tensor_key] is not None
        ):
            add_image_fn(
                visualization_dict,
                key,
                gt_decomposition[tensor_key],
                caption,
                batch_idx,
                phase,
                test_idx,
            )
    if sample is not None:
        for comp_name, comp_tensor in sample.items():
            if comp_name in [
                "rgb", "f_spec", "rgb_highlighted", "intrinsics", "supervision_mask",
            ] or not isinstance(comp_tensor, torch.Tensor):
                continue
            display_name = comp_name.replace("_", " ").title()
            add_image_fn(
                visualization_dict,
                f"images/GT_{comp_name.capitalize()}",
                comp_tensor,
                f"Ground Truth {display_name}",
                batch_idx,
                phase,
                test_idx,
            )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return visualization_dict
