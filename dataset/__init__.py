"""
Dataset module for polarization-based reflection removal and related tasks.

- UnReflectAnything_Dataset: base class for RGB (+ optional polarization) data.
- Identity defaults for named datasets live in dataset.wrappers.DATASET_DEFAULTS;
  datasets are created from config via utilities.config.create_datasets_from_config.
- Utils: adapt_intrinsics_two_step, center_crop_intrinsics, resize_intrinsics, split_videos.
"""
from .unreflectdataset import UnReflectAnything_Dataset
from .utils import (
    adapt_intrinsics_two_step,
    center_crop_intrinsics,
    resize_intrinsics,
    split_videos,
)

__all__ = [
    "UnReflectAnything_Dataset",
    "adapt_intrinsics_two_step",
    "split_videos",
    "resize_intrinsics",
    "center_crop_intrinsics",
]
