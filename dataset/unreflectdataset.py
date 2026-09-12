import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from logger import get_logger
from .polarization import PolarizationProcessor

logger = get_logger(__name__).set_context("DATASET")


class UnReflectAnything_Dataset(Dataset):
    """
    Optimized version of UnReflectAnything_Dataset dataset with performance improvements:
    1. Reduced tensor/numpy conversions
    2. Optional caching
    3. Simplified processing pipeline
    4. Batch-friendly operations
    5. Consistent image sizing for batching
    6. Support for both single file clockwise and separate polarization files
    7. Flexible scene filtering with include/exclude patterns
    8. RGB-only mode for datasets without polarization data
    9. Per-scene frame subsampling via sample_every_n (load 1/N frames)

    Polarization formats supported:
    - "single_file_clock": Single file with 4 polarization images arranged clockwise
    - "separate_files": Four separate files ending with _000, _045, _090, _135
    - "separate_files_stokes": Three separate .npy files with Stokes parameters _S0, _S1, _S2
    - "single_file_topdown": Single file with 4 polarization images arranged vertically (top to bottom)

    Scene filtering:
    - include: str or List[str] - Include scenes that match (substring or exact)
    - exclude: str or List[str] - Exclude scenes that match (substring or exact)

    RGB-only mode:
    - load_rgb_only: bool - If True, forces loading only RGB data and ignores polarization data

    File paths:
    - return_metadata: bool - If True, includes 'filepaths' dict in output with keys 'raw_path', 'pol_path', 'diffuse_path', 'intrinsics_path'
    """

    def __init__(
        self,
        root_dir: str,
        rho_s: float = 0.6,
        eps: float = 1e-8,
        rgb_ext: str = ".png",
        pol_ext: str = ".npy",
        polarization_format: str = "single_file_clock",  # "single_file_clock", "separate_files" or "mosaic"
        rgb_dir_name: str = "rgb",
        pol_dir_name: str = "pol",
        diffuse_dir_name: str = "diffuse",
        highlight_dir_name: str = "highlight",
        intrinsics_file_name: str = "intrinsics.txt",
        # Polarization data format
        # Image sizing parameters
        transform=None,
        target_size: Optional[Tuple[int, int]] = (874, 1132),  # Default size (H, W)
        resize_mode: str = "crop",  # "crop", "resize", "pad", or "resize+crop"
        # Performance optimizations
        use_cache: bool = True,
        cache_size: int = 100,
        simplify_upsampling: bool = True,  # Use simple bicubic instead of edge-aware
        precompute_stokes: bool = False,  # Precompute Stokes parameters
        # Reduced smoothing options
        smooth_specular: bool = True,
        gaussian_sigma: float = 0.0,
        dolp_min_intensity: float = 0.05,
        dolp_min_value: float = 0.04,
        # Scene filtering
        include: Optional[Union[str, List[str]]] = None,
        exclude: Optional[Union[str, List[str]]] = None,
        # Few images mode for quick testing
        few_images: bool = False,
        overfit_test: bool = False,
        # Per-folder subsampling (load 1/N images per scene folder)
        sample_every_n: int = 1,
        # RGB-only mode
        load_rgb_only: bool = False,  # Force loading only RGB data, ignore polarization
        # Return file paths in output
        return_metadata: bool = False,  # If True, include file paths in returned dictionary
        # Paired highlight images (grayscale, same name as diffuse in highlight_dir)
        load_highlight: bool = False,
        # Deprecated parameters (for backward compatibility)
        # Highlight detection (optional)
        highlight_enable: bool = False,
        highlight_brightness_threshold: float = 0.93,
        highlight_return_mask: bool = False,
        highlight_rect_size: Optional[Tuple[int, int]] = None,
        highlight_return_rect: bool = False,
        highlight_return_rect_as_rgb: bool = False,
    ):
        self.root_dir = os.path.expandvars(root_dir)
        self.rho_s = rho_s
        self.eps = eps
        self.rgb_ext = rgb_ext
        self.pol_ext = pol_ext
        self.transform = transform
        self.load_rgb_only = load_rgb_only
        self.return_metadata = return_metadata
        self.load_highlight = load_highlight
        self.rgb_dir_name = rgb_dir_name
        self.pol_dir_name = pol_dir_name
        self.diffuse_dir_name = diffuse_dir_name
        self.highlight_dir_name = highlight_dir_name
        self.intrinsics_file_name = intrinsics_file_name

        # Polarization format validation
        self.polarization_format = polarization_format.lower()
        if self.polarization_format not in [
            "single_file_clock",
            "separate_files",
            "separate_files_stokes",
            "single_file_topdown",
            "mosaic",
        ]:
            raise ValueError(
                f"polarization_format must be one of ['single_file_clock', 'separate_files', 'separate_files_stokes', 'single_file_topdown', 'mosaic'], got {polarization_format}"
            )

        # Image sizing parameters
        self.target_size = target_size
        self.resize_mode = resize_mode.lower()
        if self.resize_mode not in ["crop", "resize", "pad", "resize+crop"]:
            raise ValueError(
                f"resize_mode must be one of ['crop', 'resize', 'pad', 'resize+crop'], got {resize_mode}"
            )

        self.use_cache = use_cache
        self.simplify_upsampling = simplify_upsampling
        self.precompute_stokes = precompute_stokes

        # Simplified smoothing config
        self.smooth_specular = smooth_specular
        self.gaussian_sigma = gaussian_sigma
        self.dolp_min_intensity = dolp_min_intensity
        self.dolp_min_value = dolp_min_value

        # Store filtering parameters
        self.include = include
        self.exclude = exclude or []
        self.few_images = few_images
        # Validate and store subsampling factor
        if not isinstance(sample_every_n, int) or sample_every_n < 1:
            raise ValueError(
                f"sample_every_n must be an integer >= 1, got {sample_every_n}"
            )
        self.sample_every_n = sample_every_n

        self.scene_pairs = self._find_scene_pairs()

        # Limit to 32 samples if few_images is True
        if self.few_images and len(self.scene_pairs) > 8:
            original_count = len(self.scene_pairs)
            self.scene_pairs = self.scene_pairs[:8]
            logger.info(
                f"Few images mode: Limited dataset from {original_count} to 8 samples"
            )

        # Setup caching if enabled
        if self.use_cache:
            self._cache_intrinsics = {}
            # Make cached methods
            self._load_intrinsics_cached = lru_cache(maxsize=cache_size)(
                self._load_intrinsics_impl
            )
            if self.precompute_stokes:
                self._load_pol_cached = lru_cache(maxsize=cache_size)(
                    self._load_and_process_polarization_impl
                )
        self.overfit_test = overfit_test
        if self.overfit_test:
            self.scene_pairs = len(self.scene_pairs) * [self.scene_pairs[0]]
            logger.info(
                f"Overfit test mode: Limited dataset from {len(self.scene_pairs)} to 1 sample"
            )

        # Highlight detection configuration
        self.highlight_enabled = highlight_enable
        self.highlight_brightness_threshold = highlight_brightness_threshold
        self.highlight_return_mask = highlight_return_mask
        self.highlight_rect_size = highlight_rect_size
        self.highlight_return_rect = highlight_return_rect
        self.highlight_return_rect_as_rgb = highlight_return_rect_as_rgb

        # Polarization processor (delegates polarization loading logic)
        self._pol_processor = PolarizationProcessor(
            rho_s=self.rho_s,
            eps=self.eps,
            dolp_min_intensity=self.dolp_min_intensity,
            dolp_min_value=self.dolp_min_value,
        )

    def _resize_tensor(
        self, tensor: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Resize tensor to target size using specified mode.

        Args:
            tensor: Input tensor of shape [C, H, W] or [H, W]
            target_size: Target size as (H, W)

        Returns:
            Resized tensor of shape [C, target_H, target_W] or [target_H, target_W]

        Resize modes:
            - "crop": Center crop to target size (no resizing)
            - "resize": Resize to target size (may distort aspect ratio)
            - "pad": Pad to target size with zeros (no resizing)
            - "resize+crop": Resize to fit target size while maintaining aspect ratio, then center crop
        """
        if tensor.dim() == 2:
            # Single channel tensor [H, W]
            tensor = tensor.unsqueeze(0)  # [1, H, W]
            was_2d = True
        else:
            was_2d = False
        current_size = tensor.shape[-2:]  # [H, W]

        if self.resize_mode == "crop":
            # Center crop when larger than target; pad when smaller so output is always target_size
            if current_size[0] > target_size[0]:
                start_h = (current_size[0] - target_size[0]) // 2
                end_h = start_h + target_size[0]
                tensor = tensor[..., start_h:end_h, :]
            if current_size[1] > target_size[1]:
                start_w = (current_size[1] - target_size[1]) // 2
                end_w = start_w + target_size[1]
                tensor = tensor[..., :, start_w:end_w]
            # If any dimension is still smaller than target, pad to target_size (avoids 896x1 / 1x896)
            size_after_crop = tensor.shape[-2:]
            pad_h = max(0, target_size[0] - size_after_crop[0])
            pad_w = max(0, target_size[1] - size_after_crop[1])
            if pad_h > 0 or pad_w > 0:
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                tensor = F.pad(
                    tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=0,
                )

        elif self.resize_mode == "resize":
            # Resize to target size using bilinear interpolation
            tensor = F.interpolate(
                tensor.unsqueeze(0),  # Add batch dimension
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # Remove batch dimension

        elif self.resize_mode == "resize+crop":
            # Resize to fit target size while maintaining aspect ratio, then center crop
            target_h, target_w = target_size
            current_h, current_w = current_size
            # Guard against degenerate dimensions (avoid div-by-zero and 0-size interpolate)
            current_h = max(1, current_h)
            current_w = max(1, current_w)

            # Calculate scale factor to fit the image into target size
            scale_h = target_h / current_h
            scale_w = target_w / current_w
            scale = max(
                scale_h, scale_w
            )  # Use max to ensure we can crop to target size

            # Intermediate size must be >= target so center crop yields exactly target_size
            # (avoids [3,896,1] when int() rounds down or image has one dim very small)
            intermediate_h = max(target_h, max(1, int(current_h * scale)))
            intermediate_w = max(target_w, max(1, int(current_w * scale)))

            # Resize to intermediate size
            tensor = F.interpolate(
                tensor.unsqueeze(0),  # Add batch dimension
                size=(intermediate_h, intermediate_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # Remove batch dimension

            # Center crop to target size (valid because intermediate >= target)
            start_h = (intermediate_h - target_h) // 2
            end_h = start_h + target_h
            start_w = (intermediate_w - target_w) // 2
            end_w = start_w + target_w

            tensor = tensor[..., start_h:end_h, start_w:end_w]

        elif self.resize_mode == "pad":
            # Pad to target size with zeros
            pad_h = max(0, target_size[0] - current_size[0])
            pad_w = max(0, target_size[1] - current_size[1])

            if pad_h > 0 or pad_w > 0:
                # Pad format: (pad_left, pad_right, pad_top, pad_bottom)
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top

                tensor = F.pad(
                    tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=0,
                )

        if was_2d:
            tensor = tensor.squeeze(0)  # Remove channel dimension

        return tensor

    def _resize_raw_tensor(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Resize RAW tensor to target size.

        Args:
            raw: RAW tensor of shape [3, H, W] in [0, 1]

        Returns:
            Resized RAW tensor of shape [3, target_H, target_W] in [0, 1]
        """
        if self.target_size is None:
            return raw

        return self._resize_tensor(raw, self.target_size)

    def _resize_polarization_data(
        self, pol_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Resize all polarization data tensors to target size.

        Args:
            pol_data: Dictionary containing polarization tensors with shapes [C, H, W]
                     Keys include 'I0', 'I45', 'I90', 'I135', 'S0', 'S1', 'S2', 'DoLP', etc.

        Returns:
            Dictionary with same keys but tensors resized to target size [C, target_H, target_W]
        """
        """
        Resize all polarization data tensors to target size.

        Args:
            pol_data: Dictionary containing polarization data tensors

        Returns:
            Dictionary with all tensors resized to target size
        """
        if self.target_size is None:
            return pol_data

        resized_data = {}
        for key, tensor in pol_data.items():
            resized_data[key] = self._resize_tensor(tensor, self.target_size)
        return resized_data

    def _should_include_scene(self, scene_name: str) -> bool:
        """
        Check if a scene should be included based on include/exclude filters.

        Args:
            scene_name: Name of the scene to check

        Returns:
            True if scene should be included, False otherwise
        """
        """
        Check if a scene should be included based on include/exclude filters.

        Args:
            scene_name: Name of the scene to check

        Returns:
            True if scene should be included, False otherwise
        """
        # Check exclude filter first
        if self.exclude:
            if isinstance(self.exclude, str):
                # Single string - check if scene name contains the exclude string
                if self.exclude in scene_name:
                    return False
            elif isinstance(self.exclude, list):
                # List of strings - check exact matches and substring matches
                for exclude_pattern in self.exclude:
                    if exclude_pattern == scene_name or exclude_pattern in scene_name:
                        return False

        # Check include filter
        if self.include is not None:
            if isinstance(self.include, str):
                # Single string - check if scene name contains the include string
                return self.include in scene_name
            elif isinstance(self.include, list):
                # List of strings - check exact matches and substring matches
                for include_pattern in self.include:
                    if include_pattern == scene_name or include_pattern in scene_name:
                        return True
                # If include list is provided but no match found, exclude the scene
                return False

        # If no include filter specified and not excluded, include the scene
        return True

    def _find_scene_pairs(
        self,
    ) -> List[
        Tuple[str, Optional[str], Optional[str], Optional[str], bool, Optional[str]]
    ]:
        """
        Find matching RAW (RGB), optional diffuse, polarization, intrinsics, and optional highlight entries per scene.

        Scans the root directory structure to find corresponding files:
        - RGB images in rgb/ subdirectory
        - Polarization data in pol/ subdirectory (optional)
        - Camera intrinsics in intrinsics/ subdirectory (optional)
        - Highlight grayscale images in highlight/ subdirectory (optional, paired with rgb/diffuse by filename)

        Returns:
            List of tuples (raw_path, pol_path, diffuse_path, intrinsics_path, has_pol_data, highlight_path)
        """
        """Find valid entries based on polarization format and optional diffuse folder"""
        scene_pairs = []
        for scene_name in os.listdir(self.root_dir):
            # Use new filtering logic

            if not self._should_include_scene(scene_name):
                continue

            scene_path = os.path.join(self.root_dir, scene_name)
            if not os.path.isdir(scene_path):
                continue

            rgb_dir = os.path.join(scene_path, self.rgb_dir_name)
            pol_dir = os.path.join(scene_path, self.pol_dir_name)
            diffuse_dir = (
                os.path.join(scene_path, self.diffuse_dir_name)
                if self.diffuse_dir_name is not None
                else None
            )
            highlight_dir = (
                os.path.join(scene_path, self.highlight_dir_name)
                if self.highlight_dir_name is not None
                else None
            )
            intrinsics_path = os.path.join(scene_path, self.intrinsics_file_name)

            # Check if RGB directory exists (required)
            if not os.path.exists(rgb_dir):
                continue

            # Check if polarization directory exists (optional)
            has_pol_data = os.path.exists(pol_dir)

            # Set intrinsics_path to None if file doesn't exist
            if not os.path.exists(intrinsics_path):
                intrinsics_path = None

            rgb_files = sorted(
                [f for f in os.listdir(rgb_dir) if f.endswith(self.rgb_ext)]
            )

            # Get polarization files if pol directory exists
            pol_files = []
            if has_pol_data:
                pol_files = [f for f in os.listdir(pol_dir) if f.endswith(self.pol_ext)]
            # Get diffuse files if diffuse directory exists
            diffuse_files = []
            if diffuse_dir is not None and os.path.exists(diffuse_dir):
                diffuse_files = [
                    f for f in os.listdir(diffuse_dir) if f.endswith(self.rgb_ext)
                ]
            highlight_files = []
            if highlight_dir is not None and os.path.exists(highlight_dir):
                highlight_files = sorted(
                    [f for f in os.listdir(highlight_dir) if f.endswith(self.rgb_ext)]
                )

            # Subsample per scene folder if requested (load 1/N frames)
            if self.sample_every_n > 1:
                rgb_files = rgb_files[:: self.sample_every_n]

            filestack_iterator = (
                zip(rgb_files, highlight_files)
                if len(highlight_files) > 0
                else rgb_files
            )
            for filestack in filestack_iterator:
                rgb_file = filestack[0] if isinstance(filestack, tuple) else filestack
                highlight_file = filestack[1] if self.load_highlight else None
                raw_path = os.path.join(rgb_dir, rgb_file)
                # Determine diffuse path if available
                diffuse_path = None
                if diffuse_dir is not None and (rgb_file in diffuse_files):
                    diffuse_path = os.path.join(diffuse_dir, rgb_file)
                # Paired highlight path (grayscale, same filename as rgb/diffuse); None if not loading highlights or no match
                highlight_path = None
                if highlight_dir is not None and (highlight_file in highlight_files):
                    highlight_path = os.path.join(highlight_dir, highlight_file)
                if not has_pol_data:
                    # No polarization data available - include raw-only sample (diffuse may or may not exist)
                    scene_pairs.append(
                        (
                            raw_path,
                            None,  # No polarization path
                            diffuse_path,
                            intrinsics_path,
                            False,  # No polarization data
                            highlight_path,
                        )
                    )
                elif self.polarization_format == "single_file_clock":
                    # Original behavior: single polarization file
                    pol_file = rgb_file.replace(self.rgb_ext, self.pol_ext)
                    if pol_file in pol_files:
                        scene_pairs.append(
                            (
                                raw_path,
                                os.path.join(pol_dir, pol_file),
                                diffuse_path,
                                intrinsics_path,
                                True,  # Has polarization data
                                highlight_path,
                            )
                        )

                elif self.polarization_format == "separate_files":
                    # New behavior: separate polarization files (_000, _045, _090, _135)
                    base_name = rgb_file.replace(self.rgb_ext, "")
                    pol_files_needed = [
                        f"{base_name}_000{self.pol_ext}",
                        f"{base_name}_045{self.pol_ext}",
                        f"{base_name}_090{self.pol_ext}",
                        f"{base_name}_135{self.pol_ext}",
                    ]

                    # Check if all 4 polarization files exist
                    if all(pol_file in pol_files for pol_file in pol_files_needed):
                        # Store the base path for separate files (we'll construct individual paths later)
                        pol_base_path = os.path.join(pol_dir, base_name)
                        scene_pairs.append(
                            (
                                raw_path,
                                pol_base_path,  # Base path for separate files
                                diffuse_path,
                                intrinsics_path,
                                True,  # Has polarization data
                                highlight_path,
                            )
                        )

                elif self.polarization_format == "separate_files_stokes":
                    # New behavior: separate Stokes parameter files (_S0, _S1, _S2)
                    base_name = rgb_file.replace(self.rgb_ext, "")
                    stokes_files_needed = [
                        f"{base_name}_S0.npy",
                        f"{base_name}_S1.npy",
                        f"{base_name}_S2.npy",
                    ]

                    # Check if all 3 Stokes files exist
                    if all(
                        stokes_file in pol_files for stokes_file in stokes_files_needed
                    ):
                        # Store the base path for separate Stokes files (we'll construct individual paths later)
                        pol_base_path = os.path.join(pol_dir, base_name)
                        scene_pairs.append(
                            (
                                raw_path,
                                pol_base_path,  # Base path for separate Stokes files
                                diffuse_path,
                                intrinsics_path,
                                True,  # Has polarization data
                                highlight_path,
                            )
                        )

                elif self.polarization_format == "single_file_topdown":
                    # New behavior: single file with 4 polarization images arranged vertically
                    pol_file = rgb_file.replace(self.rgb_ext, self.pol_ext)
                    if pol_file in pol_files:
                        scene_pairs.append(
                            (
                                raw_path,
                                os.path.join(pol_dir, pol_file),
                                diffuse_path,
                                intrinsics_path,
                                True,  # Has polarization data
                                highlight_path,
                            )
                        )

        return scene_pairs

    def __len__(self) -> int:
        return len(self.scene_pairs)

    def get_loaded_scenes(self) -> List[str]:
        """
        Get list of scene names that were loaded into the dataset.

        Returns:
            List of scene names (without file extensions) that passed filtering
        """
        """Get list of scene names that are actually loaded in the dataset."""
        loaded_scenes = set()
        for raw_path, _, _, _, _, _ in self.scene_pairs:
            scene_name = os.path.basename(os.path.dirname(os.path.dirname(raw_path)))
            loaded_scenes.add(scene_name)
        return sorted(list(loaded_scenes))

    @staticmethod
    def _to_luminance_torch(rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB tensor to luminance using standard weights.

        Args:
            rgb: RGB tensor of shape [H, W, 3] in range [0, 1]

        Returns:
            Luminance tensor of shape [H, W] in range [0, 1]
        """
        """Convert RGB to luminance staying in torch. Input: [...,3] in [0,1]."""
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    def _load_intrinsics_impl(self, intrinsics_path: str) -> torch.Tensor:
        """
        Load camera intrinsics from file (implementation without caching).

        Args:
            intrinsics_path: Path to intrinsics file (typically .txt or .npy)

        Returns:
            Camera intrinsics matrix of shape [3, 3]
        """
        """Implementation for loading intrinsics (cacheable)"""
        try:
            K = np.loadtxt(intrinsics_path).reshape(3, 3).astype(np.float32)
            return torch.from_numpy(K)
        except Exception:
            # warnings.warn(f"Could not load intrinsics from {intrinsics_path}: {e}")
            return torch.eye(3, dtype=torch.float32)

    def _load_intrinsics(self, intrinsics_path: str) -> torch.Tensor:
        """
        Load camera intrinsics with optional caching.

        Args:
            intrinsics_path: Path to intrinsics file

        Returns:
            Camera intrinsics matrix of shape [3, 3]
        """
        """Load intrinsics with optional caching"""
        if self.use_cache:
            return self._load_intrinsics_cached(intrinsics_path)
        else:
            return self._load_intrinsics_impl(intrinsics_path)

    def _simple_upsample(
        self, f_spec_half: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Simple bilinear upsampling - much faster than edge-aware methods.

        Args:
            f_spec_half: Specular fraction tensor of shape [H, W] at half resolution
            target_size: Target size as (H, W) for upsampling

        Returns:
            Upsampled tensor of shape [target_H, target_W] clamped to [0, 1]
        """
        # Use torch interpolation instead of OpenCV
        f_batch = f_spec_half.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        f_up = torch.nn.functional.interpolate(
            f_batch, size=target_size, mode="bilinear", align_corners=False
        )
        return f_up.squeeze(0).squeeze(0).clamp(0, 1)

    def _load_and_process_polarization_impl(
        self, pol_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Delegates polarization loading to PolarizationProcessor. Kept as a
        method for backward compatibility and caching behavior.
        """
        return self._pol_processor.load(
            pol_path=pol_path,
            polarization_format=self.polarization_format,
            pol_ext=self.pol_ext,
        )

    def _load_single_file_polarization(self, pol_path: str) -> Dict[str, torch.Tensor]:
        """Backward-compatible wrapper to single-file loader."""
        return self._pol_processor.load_single_file_clock(pol_path)

    def _load_and_process_polarization(self, pol_path: str) -> Dict[str, torch.Tensor]:
        """Load polarization with optional caching"""
        if self.use_cache and self.precompute_stokes:
            return self._load_pol_cached(pol_path)
        else:
            return self._load_and_process_polarization_impl(pol_path)

    def _load_raw_and_separate(
        self, raw_path: str, f_spec_half: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Optimized RAW loading and separation at full resolution"""

        # Load RAW directly as torch tensor
        raw_img = Image.open(raw_path).convert("RGB")
        raw = torch.from_numpy(np.asarray(raw_img, dtype=np.float32)) / 255.0
        H, W, _ = raw.shape

        # Upsample specular fraction
        if self.simplify_upsampling:
            f_full = self._simple_upsample(f_spec_half, (H, W))
        else:
            # Fall back to original method if needed
            f_full = self._edge_aware_upsample_original(f_spec_half, raw)

        # Compute specular/diffuse separation
        raw_chw = raw.permute(2, 0, 1)  # 3xHxW
        f_expanded = f_full.unsqueeze(0)  # 1xHxW

        I_spec = (f_expanded * raw_chw).clamp(0, 1)
        I_diff = (raw_chw - I_spec).clamp(0, 1)

        # Return full resolution data - resizing will be done later if needed
        return {
            "raw": raw_chw,
            "specular": I_spec,
            "diffuse": I_diff,
        }

    def _load_raw_only(self, raw_path: str) -> Dict[str, torch.Tensor]:
        """
        Load RAW data only (when polarization data is not available) at full resolution.

        Args:
            raw_path: Path to RAW image file

        Returns:
            Dictionary containing RAW data with dummy specular/diffuse components at full resolution
        """
        # Load RAW directly as torch tensor
        raw_img = Image.open(raw_path).convert("RGB")
        raw = torch.from_numpy(np.asarray(raw_img, dtype=np.float32)) / 255.0

        # Convert to CHW format
        raw_chw = raw.permute(2, 0, 1)  # [3, H, W]

        # Create dummy specular and diffuse components (all zeros for specular, RAW for diffuse)
        I_spec = torch.zeros_like(raw_chw)  # [3, H, W] - no specular component
        I_diff = raw_chw.clone()  # [3, H, W] - all RAW is diffuse

        # Return full resolution data - resizing will be done later if needed
        return {
            "raw": raw_chw,
            "specular": I_spec,
            "diffuse": I_diff,
        }

    def _load_highlight(self, highlight_path: str) -> torch.Tensor:
        """
        Load grayscale highlight image as a 1xHxW tensor in [0, 1].

        Args:
            highlight_path: Path to grayscale image (e.g. .png in highlight_dir).

        Returns:
            Tensor of shape [1, H, W], float32 in [0, 1].
        """
        # Load as grayscale; support both single-channel and RGB (take luminance)
        img = cv2.imread(highlight_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.asarray(Image.open(highlight_path).convert("L"), dtype=np.float32)
        else:
            img = img.astype(np.float32)
        out = torch.from_numpy(img).float().div_(255.0).clamp_(0.0, 1.0)
        return out.unsqueeze(0)  # [1, H, W]

    def _load_raw_and_diffuse(
        self, raw_path: str, diffuse_path: str
    ) -> Dict[str, torch.Tensor]:
        """Load RAW and DIFFUSE images and compute SPECULAR as raw - diffuse.

        Returns a dict with keys: 'raw', 'diffuse', 'specular' as CHW tensors in [0,1].
        """
        # Load images
        raw_img = Image.open(raw_path).convert("RGB")
        raw = torch.from_numpy(np.asarray(raw_img, dtype=np.float32)) / 255.0
        diffuse_img = Image.open(diffuse_path).convert("RGB")
        diffuse = torch.from_numpy(np.asarray(diffuse_img, dtype=np.float32)) / 255.0

        # To CHW
        raw_chw = raw.permute(2, 0, 1)
        diffuse_chw = diffuse.permute(2, 0, 1)

        # Specular as residual
        specular = (raw_chw - diffuse_chw).clamp(0.0, 1.0)

        return {
            "raw": raw_chw,
            "diffuse": diffuse_chw,
            "specular": specular,
        }

    def _compute_highlight_mask(self, frame_chw: torch.Tensor) -> torch.Tensor:
        """
        Compute binary highlight mask for a single RGB frame.

        Args:
            frame_chw: Tensor of shape [C, H, W]

        Returns:
            Tensor of shape [1, H, W] with 1 at highlight pixels
        """
        if frame_chw.dim() == 3:
            if frame_chw.shape[0] == 3:
                # Use luminance weights consistent with HighlightDataset
                grayscale = (
                    0.299 * frame_chw[0] + 0.587 * frame_chw[1] + 0.114 * frame_chw[2]
                )
            else:
                grayscale = frame_chw.mean(dim=0)
        elif frame_chw.dim() == 2:
            grayscale = frame_chw
        else:
            raise ValueError(
                f"Unexpected frame dimensions for highlight mask: {frame_chw.shape}"
            )

        mask = (grayscale > self.highlight_brightness_threshold).float().unsqueeze(0)
        return mask

    def _find_rectangle_with_least_highlights(
        self, binary_mask_hw: torch.Tensor, rect_size: Optional[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Find rect (top, left, bottom, right) with fewest highlights in a binary mask.

        Args:
            binary_mask_hw: Tensor [H, W] of 0/1
            rect_size: (height, width) or None

        Returns:
            Int tensor [4] = (top, left, bottom, right)
        """
        if rect_size is None:
            return torch.tensor([0, 0, 0, 0]).int()

        target_height, target_width = rect_size
        img_height, img_width = binary_mask_hw.shape

        if target_height > img_height or target_width > img_width:
            return torch.tensor([0, 0, 0, 0]).int()

        max_top = img_height - target_height + 1
        max_left = img_width - target_width + 1

        cumsum = torch.cumsum(torch.cumsum(binary_mask_hw, dim=0), dim=1)
        padded_cumsum = F.pad(cumsum, (1, 0, 1, 0), value=0)

        tops = torch.arange(max_top, device=binary_mask_hw.device)[:, None]
        lefts = torch.arange(max_left, device=binary_mask_hw.device)[None, :]
        bottoms = tops + target_height - 1
        rights = lefts + target_width - 1

        highlight_counts = (
            padded_cumsum[bottoms + 1, rights + 1]
            - padded_cumsum[tops, rights + 1]
            - padded_cumsum[bottoms + 1, lefts]
            + padded_cumsum[tops, lefts]
        )

        min_pos = torch.argmin(highlight_counts)
        best_top = (min_pos // max_left).item()
        best_left = (min_pos % max_left).item()
        best_bottom = best_top + target_height - 1
        best_right = best_left + target_width - 1

        return torch.tensor([best_top, best_left, best_bottom, best_right]).int()

    def _crop_rectangle_from_raw(
        self, raw_chw: torch.Tensor, rect_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Crop rectangle from RAW and compute its highlight mask.

        Returns:
            (cropped_raw [C,h,w], cropped_mask [1,h,w])
        """
        top, left, bottom, right = [int(v) for v in rect_coords]

        if top == 0 and left == 0 and bottom == 0 and right == 0:
            if self.highlight_rect_size is not None:
                h, w = self.highlight_rect_size
                empty_frame = torch.zeros(
                    raw_chw.shape[0], h, w, device=raw_chw.device, dtype=raw_chw.dtype
                )
                empty_mask = torch.zeros(
                    1, h, w, device=raw_chw.device, dtype=raw_chw.dtype
                )
                return empty_frame, empty_mask
            else:
                return raw_chw, torch.zeros(
                    1,
                    raw_chw.shape[1],
                    raw_chw.shape[2],
                    device=raw_chw.device,
                    dtype=raw_chw.dtype,
                )

        cropped_raw = raw_chw[:, top : bottom + 1, left : right + 1]
        cropped_mask = self._compute_highlight_mask(cropped_raw)
        return cropped_raw, cropped_mask

    def _edge_aware_upsample_original(
        self, f_half: torch.Tensor, rgb_full: torch.Tensor
    ) -> torch.Tensor:
        """Original edge-aware upsampling (fallback)"""
        # Convert to numpy for OpenCV operations
        f_np = f_half.cpu().numpy()
        rgb_np = rgb_full.cpu().numpy()

        H, W, _ = rgb_np.shape
        f_up = cv2.resize(f_np, (W, H), interpolation=cv2.INTER_CUBIC)

        # Simple bilateral filter instead of guided filter for speed
        f_up = cv2.bilateralFilter(
            f_up.astype(np.float32), d=5, sigmaColor=0.1 * 255, sigmaSpace=5
        )

        return torch.from_numpy(np.clip(f_up, 0.0, 1.0))

    def _load_separate_polarization_files(
        self, pol_base_path: str
    ) -> Dict[str, torch.Tensor]:
        """Backward-compatible wrapper to separate-files loader."""
        return self._pol_processor.load_separate_files(pol_base_path, self.pol_ext)

    def _load_separate_stokes_files(
        self, pol_base_path: str
    ) -> Dict[str, torch.Tensor]:
        """Backward-compatible wrapper to separate Stokes loader."""
        return self._pol_processor.load_separate_stokes(pol_base_path)

    def single_arat_files_topdown(self, pol_path: str) -> Dict[str, torch.Tensor]:
        """Backward-compatible wrapper to top-down single-file loader."""
        return self._pol_processor.load_single_file_topdown(pol_path)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single dataset sample with optimized processing.

        Args:
            idx: Index of the sample to load (0 <= idx < len(dataset))

        Returns:
            Dictionary containing:
            - Polarization data: 'I0', 'I45', 'I90', 'I135', 'S0', 'S1', 'S2', 'DoLP', 'AoP', 'f_spec' (if available)
            - Image data: 'raw', 'specular', 'diffuse' (with backward-compatible alias 'rgb' == 'raw')
            - Optional 'highlight': [1, H, W] grayscale tensor (if load_highlight=True and highlight file exists)
            - Camera data: 'intrinsics' [3, 3]
            - File paths: 'filepaths' dict with keys 'raw_path', 'pol_path', 'diffuse_path', 'intrinsics_path' (if return_metadata=True)
            All image tensors have shape [C, H, W] where H, W match target_size if specified
        """
        (
            raw_path,
            pol_path,
            diffuse_path,
            intrinsics_path,
            has_pol_data,
            highlight_path,
        ) = self.scene_pairs[idx]

        # Load data (potentially from cache)
        intrinsics = self._load_intrinsics(intrinsics_path)
        if self.load_highlight and highlight_path is not None:
            sample_highlight = self._load_highlight(highlight_path)
        else:
            sample_highlight = None

        if has_pol_data and not self.load_rgb_only:
            # Load polarization data and use it for RAW separation if diffuse not provided
            pol_data = self._load_and_process_polarization(pol_path)
            # Get specular fraction for RGB processing
            f_spec = pol_data["f_spec"].squeeze(0)  # Remove batch dimension [H, W]
            if diffuse_path is not None:
                raw_data = self._load_raw_and_diffuse(raw_path, diffuse_path)
            else:
                raw_data = self._load_raw_and_separate(raw_path, f_spec)
            # Combine results
            sample = {**pol_data, **raw_data, "intrinsics": intrinsics}
        else:
            # No polarization data available or load_rgb_only is True - load RAW (and diffuse if present)
            if diffuse_path is not None:
                raw_data = self._load_raw_and_diffuse(raw_path, diffuse_path)
            else:
                raw_data = self._load_raw_only(raw_path)
            sample = {**raw_data, "intrinsics": intrinsics}
        if sample_highlight is not None:
            sample["highlight"] = sample_highlight
        elif sample_highlight is None and self.load_highlight:
            sample["highlight"] = torch.zeros_like(sample["diffuse"])[0].unsqueeze(0)
            highlight_path = "None"
        original_raw_size = sample["raw"].shape
        # Optional highlight detection and cropping on full resolution
        if self.highlight_enabled and "raw" in sample:
            raw_chw = sample["raw"]  # Full resolution RAW
            mask = self._compute_highlight_mask(raw_chw)
            if self.highlight_return_mask:
                sample["highlight_masks"] = mask
                total_pixels = raw_chw.shape[-2] * raw_chw.shape[-1]
                coverage_percent = (mask.sum() / max(total_pixels, 1)) * 100.0
                sample["highlight_coverage"] = coverage_percent.to(torch.float32)

            if self.highlight_rect_size is not None:
                rect_coords = self._find_rectangle_with_least_highlights(
                    mask.squeeze(0), self.highlight_rect_size
                )
                sample["rect_coords"] = rect_coords

                if self.highlight_return_rect:
                    rect_raw, rect_mask = self._crop_rectangle_from_raw(
                        raw_chw, rect_coords
                    )

                    sample["rect_crop"] = rect_raw
                    sample["rect_mask"] = rect_mask

                    if self.highlight_return_rect_as_rgb:
                        sample["uncropped_raw"] = sample["raw"]
                        sample["raw"] = rect_raw
                        # Also crop specular and diffuse components
                        if "specular" in sample:
                            sample["specular"] = sample["specular"][
                                :,
                                rect_coords[0] : rect_coords[2] + 1,
                                rect_coords[1] : rect_coords[3] + 1,
                            ]
                        if "diffuse" in sample:
                            sample["diffuse"] = sample["diffuse"][
                                :,
                                rect_coords[0] : rect_coords[2] + 1,
                                rect_coords[1] : rect_coords[3] + 1,
                            ]

                        # Also crop polarization data
                        pol_keys = [
                            "I0",
                            "I45",
                            "I90",
                            "I135",
                            "S0",
                            "S1",
                            "S2",
                            "S3",
                            "stokes",
                            "intensity",
                            "DoLP",
                            "AoP",
                            "AoLP",
                            "DoP",
                            "DoCP",
                            "ellipticity_angle",
                            "f_spec",
                        ]
                        for key in pol_keys:
                            if key in sample:
                                sample[key] = sample[key][
                                    :,
                                    rect_coords[0] : rect_coords[2] + 1,
                                    rect_coords[1] : rect_coords[3] + 1,
                                ]

        # Resize all data to target size if specified
        if self.target_size is not None:
            # Resize RAW-related data
            if "raw" in sample:
                sample["raw"] = self._resize_raw_tensor(sample["raw"])
            if "specular" in sample:
                sample["specular"] = self._resize_tensor(
                    sample["specular"], self.target_size
                )
            if "diffuse" in sample:
                sample["diffuse"] = self._resize_tensor(
                    sample["diffuse"], self.target_size
                )
            if "highlight" in sample:
                sample["highlight"] = self._resize_tensor(
                    sample["highlight"], self.target_size
                )
            if "highlight_masks" in sample:
                sample["highlight_masks"] = F.interpolate(
                    sample["highlight_masks"].unsqueeze(0),
                    size=self.target_size,
                    mode="nearest",
                    align_corners=None,
                ).squeeze(0)
            if "rect_crop" in sample and not self.highlight_return_rect_as_rgb:
                # Only resize rect_crop if it's not being used as the main RGB
                sample["rect_crop"] = self._resize_raw_tensor(sample["rect_crop"])
            if "rect_mask" in sample and not self.highlight_return_rect_as_rgb:
                # Only resize rect_mask if it's not being used as the main RGB
                sample["rect_mask"] = F.interpolate(
                    sample["rect_mask"].unsqueeze(0),
                    size=self.target_size,
                    mode="nearest",
                    align_corners=None,
                ).squeeze(0)

            # Resize polarization data
            pol_keys = [
                "I0",
                "I45",
                "I90",
                "I135",
                "S0",
                "S1",
                "S2",
                "S3",
                "stokes",
                "intensity",
                "DoLP",
                "AoP",
                "AoLP",
                "DoP",
                "DoCP",
                "ellipticity_angle",
                "f_spec",
            ]
            for key in pol_keys:
                if key in sample:
                    sample[key] = self._resize_tensor(sample[key], self.target_size)

        if self.transform:
            sample = self.transform(sample)

        # Add file paths if requested
        if self.return_metadata:
            metadata = {
                "raw_path": raw_path,
                "orig_size": torch.tensor(original_raw_size),
            }
            if self.load_highlight:
                metadata["highlight_path"] = highlight_path
            if not self.load_rgb_only:
                metadata["pol_path"] = pol_path
            if intrinsics_path is not None:
                metadata["intrinsics_path"] = intrinsics_path
            sample["metadata"] = metadata
        return sample
