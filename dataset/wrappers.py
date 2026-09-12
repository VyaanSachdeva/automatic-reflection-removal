"""
Dataset identity defaults live here so config/train.yaml stays minimal.

**Change defaults only in DATASET_DEFAULTS** (the dict below). Do not edit the
wrapper classes for default values—they forward DATASET_DEFAULTS automatically.
Add a new dataset by adding an entry to DATASET_DEFAULTS; a wrapper class is
optional and only needed if you set CLASS in config for custom behavior.
"""
from typing import Dict, Any
from dataset.unreflectdataset import UnReflectAnything_Dataset

# Key mapping: config uppercase -> base class kwargs (snake_case)
_IDENTITY_KEY_MAP = {
    "ROOT_DIR": "root_dir",
    "RGB_EXT": "rgb_ext",
    "POL_EXT": "pol_ext",
    "POLARIZATION_FORMAT": "polarization_format",
    "RGB_DIR_NAME": "rgb_dir_name",
    "POL_DIR_NAME": "pol_dir_name",
    "DIFFUSE_DIR_NAME": "diffuse_dir_name",
    "LOAD_HIGHLIGHT": "load_highlight",
}
_GENERIC = {
    "root_dir": "$DATASET_DIR/PLACEHOLDER/",
    "rgb_ext": ".png",
    "pol_ext": ".png",
    "polarization_format": "single_file_clock",
    "rgb_dir_name": "rgb",
    "pol_dir_name": "pol",
    "diffuse_dir_name": "diffuse",
    "load_highlight": False,
}


def _identity_kwargs_for(dataset_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Build identity kwargs from DATASET_DEFAULTS (and optional overrides). Single source of truth is the dict."""
    out = dict(_GENERIC)
    out["root_dir"] = f"$DATASET_DIR/{dataset_name}/"
    defaults = DATASET_DEFAULTS.get(dataset_name, {})
    for cfg_key, kw_key in _IDENTITY_KEY_MAP.items():
        val = (overrides or {}).get(cfg_key) if overrides is not None else None
        if val is None:
            val = defaults.get(cfg_key, out[kw_key])
        out[kw_key] = val
    return out


# Edit this dict only. Wrapper classes below read from it.
DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "SCRREAM": {
        "ROOT_DIR": "$DATASET_DIR/SCRREAM/",
        "RGB_EXT": ".png",
        "POL_EXT": ".png",
        "POLARIZATION_FORMAT": "single_file_clock",
    },
    "HOUSECAT6D": {
        "ROOT_DIR": "$DATASET_DIR/HouseCat6D/",
        "RGB_EXT": ".png",
        "POL_EXT": ".png",
        "POLARIZATION_FORMAT": "single_file_clock",
    },
    "POLARGB": {
        "ROOT_DIR": "$DATASET_DIR/PolaRGB/",
        "RGB_EXT": ".png",
        "POL_EXT": ".png",
        "POLARIZATION_FORMAT": "separate_files",
    },
    "CROMO": {
        "ROOT_DIR": "$DATASET_DIR/CroMo/",
        "RGB_DIR_NAME": "rgb/left/data",
        "POL_DIR_NAME": "polarized/left/data/",
        "RGB_EXT": ".png",
        "POL_EXT": ".png",
        "POLARIZATION_FORMAT": "single_file_topdown",
    },
    "SCARED": {
        "ROOT_DIR": "$DATASET_DIR/SCARED/",
        "RGB_EXT": ".png",
        "POL_EXT": ".png",
    },
    "STEREOMIS_TRACKING": {
        "ROOT_DIR": "$DATASET_DIR/StereoMIS_Tracking/",
        "RGB_DIR_NAME": "video_frames",
        "RGB_EXT": ".png",
    },
    "CHOLEC80": {
        "ROOT_DIR": "$DATASET_DIR/CHOLEC80/videos/",
        "RGB_DIR_NAME": "frames",
        "RGB_EXT": ".png",
    },
    "PSD": {
        "ROOT_DIR": "$DATASET_DIR/PSD_Dataset/",
        "RGB_EXT": ".png",
        "RGB_DIR_NAME": "specular",
        "DIFFUSE_DIR_NAME": "diffuse",
    },
    "SUNRGBD": {
        "ROOT_DIR": "$DATASET_DIR/SUNRGBD/",
        "RGB_DIR_NAME": "frames",
        "RGB_EXT": ".jpg",
    },
    "SCANNET": {
        "ROOT_DIR": "$DATASET_DIR/SCANNET/",
        "RGB_DIR_NAME": "rgb",
        "RGB_EXT": ".jpg",
    },
    "OPENIMAGESV7": {
        "ROOT_DIR": "$DATASET_DIR/OPENIMAGESV7/",
        "RGB_DIR_NAME": "rgb",
        "RGB_EXT": ".jpg",
    },
    "ENDOSYNTH": {
        "ROOT_DIR": "$DATASET_DIR/ENDOSYNTH/",
        "RGB_DIR_NAME": "rgb",
        "RGB_EXT": ".png",
    },
    "SHIQ": {
        "ROOT_DIR": "$DATASET_DIR/SHIQ/",
        "RGB_DIR_NAME": "diffuse",
        "HIGHLIGHT_DIR_NAME": "highlight",
        "LOAD_HIGHLIGHT": True,
        "RGB_EXT": ".png",
        
    },
    
}

# Wrapper classes (optional; only used when CLASS is set in config). They read from DATASET_DEFAULTS.
def _make_wrapper(dataset_key: str):
    """Factory: class that forwards DATASET_DEFAULTS[dataset_key] + kwargs to base."""
    class _Cls(UnReflectAnything_Dataset):
        def __init__(self, **kwargs) -> None:
            identity = _identity_kwargs_for(dataset_key)
            super().__init__(**{**identity, **kwargs})
    _Cls.__name__ = f"{dataset_key}_Dataset"
    return _Cls


SCRREAM_Dataset = _make_wrapper("SCRREAM")
HOUSECAT6D_Dataset = _make_wrapper("HOUSECAT6D")
POLARGB_Dataset = _make_wrapper("POLARGB")
CROMO_Dataset = _make_wrapper("CROMO")
SCARED_Dataset = _make_wrapper("SCARED")
STEREOMIS_TRACKING_Dataset = _make_wrapper("STEREOMIS_TRACKING")
CHOLEC80_Dataset = _make_wrapper("CHOLEC80")
PSD_Dataset = _make_wrapper("PSD")
SUNRGBD_Dataset = _make_wrapper("SUNRGBD")
SCANNET_Dataset = _make_wrapper("SCANNET")
OPENIMAGESV7_Dataset = _make_wrapper("OPENIMAGESV7")
ENDOSYNTH_Dataset = _make_wrapper("ENDOSYNTH")
SHIQ_Dataset = _make_wrapper("SHIQ")