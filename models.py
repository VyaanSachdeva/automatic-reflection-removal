import importlib
import inspect
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

# Optional logging: only imports logger when a log is emitted (avoids pulling in
# dev pipeline when using endpoints/model_.py for inference-only).
def _log_info(msg: str, context: str = "MODEL") -> None:
    try:
        from logger import get_logger
        get_logger(__name__).set_context(context).info(msg)
    except ImportError:
        pass

def _log_warning(msg: str, context: str = "MODEL") -> None:
    try:
        from logger import get_logger
        get_logger(__name__).set_context(context).warning(msg)
    except ImportError:
        pass

def _build(component, cls):
    """Build component from instance or config dict (no dependency on utilities.model)."""
    if isinstance(component, cls):
        return component
    if isinstance(component, dict):
        return cls(component)
    raise TypeError(f"Expected {cls.__name__} instance or dict config, got {type(component)}.")

def pixel_mask_to_patch_mask(
    mask_hw: torch.Tensor,
    patch_size: int,
    threshold: float = 0.0,
    invert: bool = False,
    soft: bool = False,
) -> torch.Tensor:
    """Convert pixel mask (B,1,H,W) to patch mask (B,N). Uses only torch."""
    if mask_hw.dtype == torch.bool:
        m = mask_hw.float()
    else:
        m = (mask_hw > threshold).float()
    if soft:
        m_small = F.avg_pool2d(m, kernel_size=patch_size, stride=patch_size)
        pm = m_small.flatten(1)
    else:
        m_small = F.max_pool2d(m, kernel_size=patch_size, stride=patch_size)
        pm = m_small.flatten(1).bool()
    if invert:
        if soft:
            pm = 1.0 - pm
        else:
            pm = torch.logical_not(pm)
    return pm

def feather_token_mask(pm_soft: torch.Tensor, radius_tokens: int = 1, smoothstep: bool = True) -> torch.Tensor:
    """Feather token mask (B,N) over the token grid. Uses only torch."""
    B, N = pm_soft.shape
    H = int(round(N**0.5))
    assert H * H == N, "N must be a perfect square (flattened token grid)."
    m = pm_soft.view(B, 1, H, H)
    k = 2 * radius_tokens + 1
    pad = radius_tokens
    kernel = torch.ones(1, 1, k, k, device=m.device, dtype=m.dtype) / (k * k)
    m_blur = F.conv2d(m, kernel, padding=pad)
    inv_blur = F.conv2d(1.0 - m, kernel, padding=pad)
    out = m_blur / (m_blur + inv_blur + 1e-6)
    if smoothstep:
        out = out * out * (3.0 - 2.0 * out)
    return out.clamp_(0, 1).view(B, N)


@torch.no_grad()
def detect_sky_mask(
    rgb: torch.Tensor,
    blue_excess: float = 0.02,
    min_brightness: float = 0.45,
    max_gradient: float = 0.06,
    top_fraction: float = 0.6,
    open_kernel: int = 9,
    dilate_kernel: int = 0,
    white_sky: bool = False,
    white_brightness: float = 0.85,
) -> torch.Tensor:
    """
    Very simple, dependency-free sky detector for RGB images in [0,1], (B,3,H,W).

    Sky is told apart from specular highlights by three cues that highlights do
    NOT share:
      - blue-dominant colour (highlights are achromatic/white),
      - low local gradient / smoothness (highlights are small & high-contrast),
      - location in the upper part of the frame.
    A morphological opening drops small bright specks so isolated highlights are
    never picked up. The blueness gate keeps this from firing on red/indoor/
    endoscopic scenes or on bright *white* synthetic training highlights, so it
    is safe to run during training as well as inference. `white_sky` optionally
    adds an overcast/white-sky branch (off by default; can clash with white
    synthetic highlights during training).

    Returns a float mask (B,1,H,W), 1 = sky.
    """
    B, C, Hh, Ww = rgb.shape
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    blueness = b - torch.maximum(r, g)

    # Sobel gradient magnitude of luma as a flatness cue.
    kx = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=rgb.device, dtype=rgb.dtype
    ).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=rgb.device, dtype=rgb.dtype
    ).view(1, 1, 3, 3) / 8.0
    lp = F.pad(luma, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(lp, kx)
    gy = F.conv2d(lp, ky)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)

    flat = grad < max_gradient
    blue = blueness > blue_excess
    bright = luma > min_brightness
    rows = torch.linspace(0, 1, Hh, device=rgb.device, dtype=rgb.dtype).view(
        1, 1, Hh, 1
    )
    upper = rows < top_fraction

    sky = flat & upper & blue & bright
    if white_sky:
        sky = sky | (flat & upper & (luma > white_brightness))
    sky = sky.to(rgb.dtype)

    # Morphological opening (erode then dilate) removes small flat-bright specks,
    # keeping only large coherent sky regions.
    if open_kernel and open_kernel > 1:
        pad = open_kernel // 2
        eroded = -F.max_pool2d(-sky, open_kernel, 1, pad)
        sky = F.max_pool2d(eroded, open_kernel, 1, pad)
    if dilate_kernel and dilate_kernel > 1:
        pad = dilate_kernel // 2
        sky = F.max_pool2d(sky, dilate_kernel, 1, pad)
    return sky


class DINOv3(nn.Module):
    """
    Configurable DINOv3 model with flexible return optionas.
    Supports returning hidden states and feature maps in various formats.
    """

    def __init__(self, config):
        """
        Initialize DINOv3 model with configuration.

        Args:
            config: dict containing:
                - model_name: str, DINOv3 model name (default: "facebook/dinov3-vitb16-pretrain-lvd1689m")
                - image_size: int, input image size (default: 896). The input image will be preprocessed to have this size (square)
                - freeze_backbone: bool, whether to freeze DINOv3 parameters (default: True)
                - return_last_hidden_state: bool, return last hidden state (default: True)
                - return_all_hidden_states: bool, return all hidden states (default: False)
                - return_selected_layers: list[int], specific layer indices to return (default: None)
                - return_as_feature_maps: bool, reshape patch tokens to spatial format (default: False)
                - return_cls_token: bool, return CLS token (default: False)
                - return_register_tokens: bool, return register tokens (default: False)
        """
        super().__init__()

        # Configuration with defaults
        self.config = {
            "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "image_size": 896,
            "freeze_backbone": True,
            "return_last_hidden_state": True,
            "return_all_hidden_states": False,
            "return_selected_layers": None,
            "return_patch_tokens_only": True,
            "return_as_feature_maps": False,
            "return_cls_token": False,
            "return_register_tokens": False,
            **config,  # Override defaults with user config
        }

        # DINOv3 backbone
        self.dinov3 = AutoModel.from_pretrained(self.config["model_name"])
        self.processor = AutoImageProcessor.from_pretrained(self.config["model_name"])
        self.processor.size = {
            "height": self.config["image_size"],
            "width": self.config["image_size"],
        }

        # Freeze parameters if requested
        if self.config["freeze_backbone"]:
            for param in self.dinov3.parameters():
                param.requires_grad = False

        # Model properties
        self.feature_dim = self.dinov3.config.hidden_size  # 768 for ViT-B/16
        self.patch_size = self.dinov3.config.patch_size  # 16 for DINOv3
        self.dinov3.config.image_size = self.config["image_size"]

    def get_patch_spatial_dims(self, input_height, input_width):
        """Calculate spatial dimensions of patch features based on input size."""
        patch_h = input_height // self.patch_size
        patch_w = input_width // self.patch_size
        return patch_h, patch_w

    def extract_cls_token(self, hidden_states):
        """
        Extract CLS token from hidden states.

        Args:
            hidden_states: [B, N_tokens, feature_dim] - Raw hidden states from DINOv3

        Returns:
            [B, feature_dim] - CLS token features
        """
        return hidden_states[:, 0]  # CLS token is at index 0

    def extract_register_tokens(self, hidden_states):
        """
        Extract register tokens from hidden states.

        Args:
            hidden_states: [B, N_tokens, feature_dim] - Raw hidden states from DINOv3

        Returns:
            [B, 4, feature_dim] - Register token features (4 register tokens)
        """
        return hidden_states[:, 1:5]  # Register tokens are at indices 1-4

    def tokens_to_feature_maps(self, hidden_states, batch_size, patch_h, patch_w):
        """
        Convert patch tokens to spatial feature maps.

        Args:
            hidden_states: [B, N_tokens, feature_dim] - Raw hidden states from DINOv3
            batch_size: int - Batch size
            patch_h, patch_w: int - Spatial dimensions of patches

        Returns:
            [B, feature_dim, patch_h, patch_w] - Spatial feature maps
        """
        # Remove CLS token (index 0) and register tokens (indices 1-4)
        # DINOv3 has 1 CLS token + 4 register tokens + patch tokens
        patch_tokens = hidden_states[:, 5:]  # [B, N_patches, feature_dim]

        # Reshape to spatial format
        # Ensure contiguous layout after transpose for safe view/grad layout under DDP
        patch_tokens = patch_tokens.transpose(
            1, 2
        ).contiguous()  # [B, feature_dim, N_patches]
        feature_maps = patch_tokens.view(batch_size, self.feature_dim, patch_h, patch_w)

        return feature_maps

    def forward(self, rgb_image):
        """
        Forward pass with configurable return options.

        Args:
            rgb_image: [B, 3, H, W] - Input RGB images (should be preprocessed for DINOv3)
                      Can be any size as long as H and W are divisible by patch_size

        Returns:
            dict containing requested outputs based on config:
                - 'last_hidden_state': [B, N_tokens, feature_dim] or [B, feature_dim, patch_h, patch_w]
                - 'cls_token': [B, feature_dim] - CLS token features
                - 'register_tokens': [B, 4, feature_dim] - Register token features (4 tokens)
                - 'all_hidden_states': List of [B, N_tokens, feature_dim] or [B, feature_dim, patch_h, patch_w]
                - 'selected_hidden_states': List of selected layer outputs
        """
        batch_size, _, input_h, input_w = rgb_image.shape

        # Ensure input dimensions are compatible with patch size
        assert input_h % self.patch_size == 0, (
            f"Height {input_h} must be divisible by patch size {self.patch_size}"
        )
        assert input_w % self.patch_size == 0, (
            f"Width {input_w} must be divisible by patch size {self.patch_size}"
        )

        # Calculate patch spatial dimensions
        patch_h, patch_w = self.get_patch_spatial_dims(input_h, input_w)

        # Get DINOv3 outputs
        need_all_hidden_states = (
            self.config["return_all_hidden_states"]
            or self.config["return_selected_layers"] is not None
        )

        outputs = self.dinov3(rgb_image, output_hidden_states=need_all_hidden_states)

        # Prepare return dictionary
        result = {}

        # Return last hidden state
        if self.config["return_last_hidden_state"]:
            last_hidden = outputs.last_hidden_state  # [B, N_tokens, feature_dim]
            if self.config["return_as_feature_maps"]:
                last_hidden = self.tokens_to_feature_maps(
                    last_hidden, batch_size, patch_h, patch_w
                )
            result["last_hidden_state"] = last_hidden

        # Return CLS token
        if self.config["return_cls_token"]:
            cls_token = self.extract_cls_token(
                outputs.last_hidden_state
            )  # [B, feature_dim]
            result["cls_token"] = cls_token

        # Return register tokens
        if self.config["return_register_tokens"]:
            register_tokens = self.extract_register_tokens(
                outputs.last_hidden_state
            )  # [B, 4, feature_dim]
            result["register_tokens"] = register_tokens

        # Return all hidden states
        if self.config["return_all_hidden_states"]:
            all_hidden = outputs.hidden_states  # Tuple of [B, N_tokens, feature_dim]
            if self.config["return_patch_tokens_only"]:
                all_hidden = [h[:, 5:] for h in all_hidden]
            if self.config["return_as_feature_maps"]:
                all_hidden = [
                    self.tokens_to_feature_maps(h, batch_size, patch_h, patch_w)
                    for h in all_hidden
                ]
            result["all_hidden_states"] = all_hidden

        # Return selected hidden states
        if self.config["return_selected_layers"] is not None:
            selected_layers = self.config["return_selected_layers"]
            all_hidden = outputs.hidden_states
            if self.config["return_patch_tokens_only"]:
                all_hidden = [h[:, 5:] for h in all_hidden]
            selected_hidden = [all_hidden[i] for i in selected_layers]
            if self.config["return_as_feature_maps"]:
                selected_hidden = [
                    self.tokens_to_feature_maps(h, batch_size, patch_h, patch_w)
                    for h in selected_hidden
                ]
            result["selected_hidden_states"] = selected_hidden

        return result

    def preprocess_image(self, image_tensor):
        """
        Preprocess image for DINOv3 using the proper processor.

        Args:
            image_tensor: [B, 3, H, W] - Raw image tensor with values in [0, 1]

        Returns:
            [B, 3, H, W] - Preprocessed tensor ready for DINOv3
        """
        import PIL.Image

        # Vectorized conversion to [0, 255] range and permute dimensions
        # [B, 3, H, W] -> [B, H, W, 3]
        img_batch = (image_tensor * 255).byte().permute(0, 2, 3, 1)  # [B, H, W, 3]

        # Convert entire batch to numpy for PIL processing
        img_numpy = img_batch.cpu().numpy()  # [B, H, W, 3]

        # Create PIL images from the entire batch
        pil_images = [
            PIL.Image.fromarray(img_numpy[i], mode="RGB")
            for i in range(img_numpy.shape[0])
        ]

        # Process entire batch at once using the processor
        processed = self.processor(images=pil_images, return_tensors="pt")
        processed_batch = processed["pixel_values"]  # [B, 3, H, W]

        return processed_batch.to(image_tensor.device)


class DPTReassembleLayer(nn.Module):
    """
    Reassemble layer to convert transformer tokens to spatial feature maps.
    Handles projection, spatial rearrangement, and upsampling/downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float,
        readout_type: str = "ignore",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.readout_type = readout_type

        # Readout projection if using "project" method
        if readout_type == "project":
            self.readout_project = nn.Sequential(
                nn.Linear(2 * in_channels, in_channels), nn.GELU()
            )

        # Channel projection
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Spatial resampling based on scale factor
        if scale_factor == 4.0:
            # 4x upsampling: 24x24 -> 96x96
            self.resample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=8,
                stride=4,
                padding=2,
                bias=True,
            )
        elif scale_factor == 2.0:
            # 2x upsampling: 24x24 -> 48x48
            self.resample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            )
        elif scale_factor == 1.0:
            # No resampling: 24x24 -> 24x24
            self.resample = nn.Identity()
        elif scale_factor == 0.5:
            # 0.5x downsampling: 24x24 -> 12x12
            self.resample = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            )

    def forward(
        self, hidden_state: torch.Tensor, patch_h: int, patch_w: int
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, N_tokens, feature_dim] - includes CLS + register + patch tokens
            patch_h, patch_w: Spatial dimensions of patches
        Returns:
            [B, out_channels, H', W'] - Spatial feature map
        """
        batch_size = hidden_state.shape[0]

        # Reshape to spatial format
        # Make tensor contiguous after transpose so the 4D view has standard strides
        patch_tokens = hidden_state.transpose(
            1, 2
        ).contiguous()  # [B, feature_dim, patch_h*patch_w]
        patch_tokens = patch_tokens.view(
            batch_size, self.in_channels, patch_h, patch_w
        )  # [B, feature_dim, patch_h, patch_w]

        # Project channels
        patch_tokens = self.proj(patch_tokens)  # [B, out_channels, patch_h, patch_w]

        # Resample spatial resolution
        output = self.resample(patch_tokens)  # [B, out_channels, H', W']

        return output


class DPTFeatureFusionBlock(nn.Module):
    """
    Feature fusion block based on RefineNet architecture.
    Combines features from different scales using residual connections.
    """

    def __init__(self, in_channels: int, out_channels: int = 256, use_bn: bool = False):
        super().__init__()

        self.residual_conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=not use_bn
        )
        self.residual_conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn
            ),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn
            ),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
        )

        self.relu = nn.ReLU(inplace=True)

        # Output projection
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
        Returns:
            [B, out_channels, H, W]
        """
        # Project to output channels
        residual = self.residual_conv1(x)  # [B, out_channels, H, W]

        # Residual refinement
        out = self.residual_conv2(residual)  # [B, out_channels, H, W]
        out = self.relu(out + residual)  # [B, out_channels, H, W]

        # Final projection
        out = self.out_conv(out)  # [B, out_channels, H, W]

        return out


class DPT_Decoder(nn.Module):
    """
    DPT decoder adapted for RGB output from DINOv3 features.
    Implements multi-scale reassembly, progressive fusion, and RGB prediction head.

    Dropout:
        Controlled by config["dropout"] (float in [0,1]). Applied as 2D dropout
        after each fusion addition and between stages in the head to regularize
        training. Shapes are preserved; tensor dims are:
        - inputs hidden_states: List[4] of [B, N_tokens, C]
        - output: [B, C_out, H, W]
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        # Default configuration
        default_config = {
            "feature_dim": 768,  # DINOv3 ViT-B feature dimension
            "reassemble_out_channels": [96, 192, 384, 768],  # Neck hidden sizes
            "reassemble_factors": [4.0, 2.0, 1.0, 0.5],  # Spatial scale factors
            "fusion_hidden_size": 256,
            "readout_type": "ignore",  # 'ignore', 'add', or 'project'
            "use_bn": False,
            "dropout": 0.0,
            "output_image_size": (448, 448),  # If None, maintains input size
            "output_channels": 3,  # Set to 4 for RGBA output
        }

        self.config = {**default_config, **(config or {})}
        self.out_image_size = self.config["output_image_size"]
        # Create reassemble layers for multi-scale feature extraction
        self.reassemble_layers = nn.ModuleList(
            [
                DPTReassembleLayer(
                    in_channels=self.config["feature_dim"],
                    out_channels=out_ch,
                    scale_factor=scale,
                    readout_type=self.config["readout_type"],
                )
                for out_ch, scale in zip(
                    self.config["reassemble_out_channels"],
                    self.config["reassemble_factors"],
                )
            ]
        )

        # Create fusion blocks for progressive feature combination
        fusion_in_channels = self.config["reassemble_out_channels"]
        self.fusion_blocks = nn.ModuleList(
            [
                DPTFeatureFusionBlock(
                    in_channels=ch,
                    out_channels=self.config["fusion_hidden_size"],
                    use_bn=self.config["use_bn"],
                )
                for ch in fusion_in_channels
            ]
        )

        # Dropout used after fusion additions and inside head stages
        p = float(self.config.get("dropout", 0.0))
        self.drop2d = nn.Dropout2d(p) if p and p > 0.0 else nn.Identity()

        # RGB prediction head
        self.rgb_head = nn.Sequential(
            # First stage: 256 -> 128 channels with spatial refinement
            nn.Conv2d(self.config["fusion_hidden_size"], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # 192x192 -> 384x384
            # Second stage: 128 -> 64 channels with feature refinement
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
            # Third stage: 64 -> 32 channels
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
            # Final RGB projection
            nn.Conv2d(32, self.config["output_channels"], kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(
        self,
        hidden_states: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through DPT decoder.

        Args:
            hidden_states: List of 4 tensors from DINOv3 selected layers
                          Each tensor: [B, N_tokens, feature_dim] where N_tokens = 5 + (H/16) * (W/16)
            input_height: Original input image height
            input_width: Original input image width

        Returns:
            rgb_output: [B, 3, H, W] - RGB image in [0, 1] range
        """
        input_height, input_width = self.out_image_size
        # Calculate patch grid dimensions
        patch_h = input_height // 16  # DINOv3 uses patch_size=16
        patch_w = input_width // 16
        # Apply reassemble layers to create multi-scale feature maps
        reassembled_features = []
        for i, (hidden_state, reassemble) in enumerate(
            zip(hidden_states, self.reassemble_layers)
        ):
            feature_map = reassemble(hidden_state, patch_h, patch_w)
            reassembled_features.append(feature_map)

        # Expected spatial dimensions after reassembly (for 384x384 input):
        # Stage 0: [B, 96, 96, 96]   (4x upsampling from 24x24)
        # Stage 1: [B, 192, 48, 48]  (2x upsampling from 24x24)
        # Stage 2: [B, 384, 24, 24]  (no resampling)
        # Stage 3: [B, 768, 12, 12]  (0.5x downsampling from 24x24)

        # Progressive fusion from smallest to largest scale
        # Start with smallest scale (stage 3)
        fused = self.fusion_blocks[3](reassembled_features[3])  # [B, 256, 12, 12]
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # [B, 256, 24, 24]

        # Add stage 2 features
        fused = fused + self.fusion_blocks[2](
            reassembled_features[2]
        )  # [B, 256, 24, 24]
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # [B, 256, 48, 48]

        # Add stage 1 features
        fused = fused + self.fusion_blocks[1](
            reassembled_features[1]
        )  # [B, 256, 48, 48]
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # [B, 256, 96, 96]

        # Add stage 0 features
        fused = fused + self.fusion_blocks[0](
            reassembled_features[0]
        )  # [B, 256, 96, 96]
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # [B, 256, 192, 192]

        # Apply RGB head
        rgb_output = self.rgb_head(fused)  # [B, 3, 384, 384]

        # Resize to original input size if needed
        if self.config["output_image_size"] or (
            rgb_output.shape[-2:] != (input_height, input_width)
        ):
            target_size = self.config["output_image_size"] or (
                input_height,
                input_width,
            )
            rgb_output = F.interpolate(
                rgb_output, size=target_size, mode="bilinear", align_corners=True
            )

        return rgb_output


class UnReflect_Model(nn.Module):
    """
    RGB with flexible DPT decoders.

    Inputs (forward):
      batch["rgb"] : (B,3,H,W) in [0,1]

    Returns:
      {
        "decoder_name": (B,C,H,W) for each configured decoder
        "tokens": {
           "rgb": (B,N,C),
        }
      }
    """

    def __init__(
        self,
        # 1) RGB encoder (DINOv3) — instance or config dict
        dinov3,
        # 2) Flexible decoders — dict of decoder_name -> DPT_Decoder config/instance
        decoders=None,  # Dict[str, DPT_Decoder instance or dict config]
        # Legacy support for backward compatibility (will be deprecated)
        spec_decoder=None,  # DPT_Decoder instance or dict
        diffuse_decoder=None,  # DPT_Decoder instance or dict
        highlight_decoder=None,  # DPT_Decoder instance or dict
        # Optional: if your DINO wrapper needs these hints
        patch_size: int = 16,
        **kwargs,
    ):
        super().__init__()

        # ---- Detect ConvNeXt from encoder config ----
        # Check if encoder name contains "convnext" (case-insensitive)
        is_convnext = False
        if isinstance(dinov3, dict):
            model_name = dinov3.get("model_name", "").lower()
            is_convnext = "convnext" in model_name
        else:
            pass
        # DEPRECATED: DINOv3_ConvNext was never used
        # elif isinstance(dinov3, (DINOv3,DINOv3_ConvNext)):
        #     # If it's already an instance, check its type
        #     is_convnext = isinstance(dinov3, DINOv3_ConvNext)
        #     if isinstance(dinov3, DINOv3):
        #         model_name = dinov3.config.get("model_name", "").lower()
        #         is_convnext = "convnext" in model_name

        # Extract encoder learning rate before building encoder
        encoder_lr = None
        if isinstance(dinov3, dict):
            encoder_lr = dinov3.get("encoder_lr", dinov3.get("RGB_ENCODER_LR", None))

        # Accept either an instance or a DINOv3(**cfg) / DINOv3_ConvNext(**cfg) dict
        if not is_convnext:
            self.dinov3 = _build(dinov3, DINOv3)
            self.use_convnext = False
        else:
            pass
            # DEPRECATED: DINOv3_ConvNext was never used
            # self.dinov3 = _build(dinov3, DINOv3_ConvNext)
            # self.use_convnext = True

        # Freeze encoder if encoder_lr is explicitly 0.0
        if encoder_lr is not None and encoder_lr == 0.0:
            for param in self.dinov3.parameters():
                param.requires_grad = False
            self.dinov3.eval()
            if kwargs.get("verbose", False):
                _log_info("RGB Encoder frozen due to RGB_ENCODER_LR=0.0")
        else:
            # Ensure encoder is trainable if LR is set (or None, which will be handled in optimizer)
            self.dinov3.train()
            if encoder_lr is not None and kwargs.get("verbose", False):
                _log_info(f"RGB Encoder trainable with RGB_ENCODER_LR={encoder_lr}")

        # Store encoder learning rate for optimizer setup
        self.encoder_lr = encoder_lr

        self.image_size = self.dinov3.config["image_size"]
        self.patch_size = patch_size
        self.embed_dim = self.dinov3.feature_dim

        # For ConvNeXt, try to detect per-stage feature dimensions from model config
        convnext_feature_dims = None
        if self.use_convnext:
            # Try to get feature dimensions from ConvNeXt model config
            if hasattr(self.dinov3.dinov3, "config"):
                model_config = self.dinov3.dinov3.config
                # ConvNeXt models typically have hidden_sizes in config
                if hasattr(model_config, "hidden_sizes"):
                    all_feature_dims = list(model_config.hidden_sizes)
                    # Get selected layers to match feature dimensions
                    selected_layers = self.dinov3.config.get(
                        "return_selected_layers", None
                    )
                    if selected_layers is not None and len(selected_layers) > 0:
                        # Map selected layer indices to feature dimensions
                        # Note: hidden_sizes might include initial embedding, so we need to map correctly
                        # For ConvNeXt, hidden_sizes typically corresponds to stage outputs
                        # If we have more stages than selected layers, we need to map correctly
                        if len(all_feature_dims) >= len(selected_layers):
                            # Try to map selected layers to feature dimensions
                            # This is a heuristic - may need adjustment based on actual model structure
                            max_layer = max(selected_layers)
                            if max_layer < len(all_feature_dims):
                                # Use feature dimensions corresponding to selected layers
                                convnext_feature_dims = [
                                    all_feature_dims[i] for i in selected_layers
                                ]
                            else:
                                # Fallback: use last N dimensions
                                convnext_feature_dims = all_feature_dims[
                                    -len(selected_layers) :
                                ]
                        else:
                            # Not enough stages, use what we have
                            convnext_feature_dims = all_feature_dims
                    else:
                        # No selected layers specified, use last 4 stages
                        if len(all_feature_dims) >= 4:
                            convnext_feature_dims = all_feature_dims[-4:]
                        else:
                            convnext_feature_dims = all_feature_dims
                    # if convnext_feature_dims:
                    #     _log_info(
                    #         f"ConvNeXt feature dimensions: {convnext_feature_dims}"
                    #     )

        # ---- Decoders (DPT_Decoder or DPT_Decoder_ConvNext) ----
        # Handle flexible decoder configuration with legacy support
        def build_dpt(dec, decoder_name=None):
            """Build a decoder from config or instance (standard, FiLM-conditioned, or ConvNext).

            Accepts either an instantiated `DPT_Decoder`/`DPT_Decoder_ConvNext`/`FiLMConditionedDPT` or a
            config dict. When a config dict is provided, if it contains
            `use_film` (or `USE_FILM`) set to True, a `FiLMConditionedDPT` is
            created; otherwise a standard `DPT_Decoder` or `DPT_Decoder_ConvNext` is created
            based on whether we're using ConvNeXt encoder. The config
            dict is passed as a single dictionary argument to the decoder's
            constructor, augmented with the resolved `feature_dim`.

            If `from_pretrained` (or `FROM_PRETRAINED`) is set and not empty,
            the decoder weights are loaded from that path and the decoder is frozen.

            Args:
                dec: Decoder instance or config dict
                decoder_name: Optional decoder name for prefix stripping when loading weights
            """
            # DEPRECATED: DINOv3_ConvNext or FiLM confitioning were never used

            # if isinstance(dec, (DPT_Decoder, DPT_Decoder_ConvNext, FiLMConditionedDPT)):
            #     return dec
            if isinstance(dec, dict):
                # Extract pretrained path before building decoder
                pretrained_relative_path = dec.get(
                    "from_pretrained", dec.get("FROM_PRETRAINED", "")
                )
                if pretrained_relative_path != "":
                    pretrained_path = os.path.join(
                        os.environ.get("WEIGHTS_DIR", "weights"),
                        pretrained_relative_path,
                    )
                else:
                    pretrained_path = None
                # Extract decoder learning rate
                decoder_lr = dec.get("decoder_lr", dec.get("DECODER_LR", None))
                # Extract selective freezing parameters (before building config)
                num_fusion_blocks_trainable = dec.get(
                    "num_fusion_blocks_trainable",
                    dec.get("NUM_FUSION_BLOCKS_TRAINABLE", None),
                )
                train_rgb_head = dec.get(
                    "train_rgb_head", dec.get("TRAIN_RGB_HEAD", None)
                )
                # Determine whether to build FiLM-conditioned or standard decoder
                # Build config dict for the decoder class
                # For ConvNeXt, use detected feature dimensions if available, otherwise use embed_dim
                # But allow explicit feature_dim in decoder config to override auto-detection
                if "feature_dim" in dec or "FEATURE_DIM" in dec:
                    # User explicitly set feature_dim, use it
                    feature_dim = dec.get(
                        "feature_dim", dec.get("FEATURE_DIM", self.embed_dim)
                    )
                elif is_convnext and convnext_feature_dims is not None:
                    # Use auto-detected dimensions
                    feature_dim = convnext_feature_dims
                else:
                    # Fallback to embed_dim
                    feature_dim = self.embed_dim
                config = {
                    "feature_dim": feature_dim,
                    **dec,
                }
                # Remove the control flags from the config passed into the module
                config.pop("use_film", None)
                config.pop("USE_FILM", None)
                config.pop("from_pretrained", None)
                config.pop("FROM_PRETRAINED", None)
                config.pop("DECODER_LR", None)
                config.pop("num_fusion_blocks_trainable", None)
                config.pop("NUM_FUSION_BLOCKS_TRAINABLE", None)
                config.pop("train_rgb_head", None)
                config.pop("TRAIN_RGB_HEAD", None)

                # Create decoder instance - use ConvNext decoder if using ConvNext encoder
                decoder = DPT_Decoder(config)
                # DEPRECATED: DINOv3_ConvNext or FiLM confitioning were never used
                # if use_film:
                #     decoder = FiLMConditionedDPT(config)
                # elif is_convnext:
                #     decoder = DPT_Decoder_ConvNext(config)

                # Load pretrained weights and freeze if path is specified and not empty
                if pretrained_path and pretrained_path != "":
                    if not os.path.exists(pretrained_path):
                        raise FileNotFoundError(
                            f"Pretrained decoder weights not found at: {pretrained_path}"
                        )
                    # Load checkpoint (handle both raw state_dict and checkpoint formats)
                    if not os.path.exists(pretrained_path):
                        # Try swapping .pth <-> .pt if file not found
                        if pretrained_path.endswith(".pt"):
                            alt_pretrained_path = pretrained_path[:-3] + ".pth"
                        elif pretrained_path.endswith(".pth"):
                            alt_pretrained_path = pretrained_path[:-4] + ".pt"
                        else:
                            alt_pretrained_path = None

                        if alt_pretrained_path and os.path.exists(alt_pretrained_path):
                            pretrained_path = alt_pretrained_path
                        else:
                            return None

                    checkpoint = torch.load(
                        pretrained_path, map_location="cpu", weights_only=False
                    )

                    # Extract state dict (handle both formats)
                    state_dict = checkpoint
                    if isinstance(checkpoint, dict):
                        # Try common checkpoint keys
                        state_dict = (
                            checkpoint.get("model_state_dict")
                            or checkpoint.get("state_dict")
                            or checkpoint
                        )

                    # Strip common prefixes if state dict was saved as part of larger model
                    # Handle cases like "decoders.diffuse.weight" -> "weight" or "decoder.weight" -> "weight"
                    if (
                        isinstance(state_dict, dict)
                        and decoder_name
                        and len(state_dict) > 0
                    ):
                        sample_key = next(iter(state_dict.keys()))
                        if "." in sample_key:
                            # Try to find decoder-specific prefix pattern
                            prefix_options = [
                                f"decoders.{decoder_name}.",
                                f"{decoder_name}.",
                                "decoder.",
                            ]
                            for prefix in prefix_options:
                                # Check if any keys start with this prefix
                                matching_keys = [
                                    k for k in state_dict.keys() if k.startswith(prefix)
                                ]
                                if matching_keys:
                                    # Strip decoder prefix from matching keys, keep others as-is
                                    stripped_dict = {}
                                    for k, v in state_dict.items():
                                        if k.startswith(prefix):
                                            stripped_dict[k[len(prefix) :]] = v
                                        else:
                                            stripped_dict[k] = v
                                    state_dict = stripped_dict
                                    break

                    # Load weights with strict=False to handle partial matches
                    # Filter out keys with shape mismatches before loading
                    filtered_state_dict = {}
                    incompatible_keys = []
                    model_state_dict = decoder.state_dict()

                    for key, value in state_dict.items():
                        if key in model_state_dict:
                            model_shape = model_state_dict[key].shape
                            checkpoint_shape = value.shape
                            if model_shape == checkpoint_shape:
                                filtered_state_dict[key] = value
                            else:
                                incompatible_keys.append(
                                    f"{key}: checkpoint shape {checkpoint_shape} != model shape {model_shape}"
                                )
                        else:
                            # Key not in model, skip it
                            pass

                    # Load filtered state dict
                    missing_keys, unexpected_keys = decoder.load_state_dict(
                        filtered_state_dict, strict=True
                    )

                    # Print confirmation about weight loading status
                    total_model_keys = len(model_state_dict)
                    loaded_keys = len(filtered_state_dict)
                    has_issues = bool(
                        incompatible_keys or missing_keys or unexpected_keys
                    )

                    if not has_issues and loaded_keys == total_model_keys:
                        if kwargs.get("verbose", False):
                            _log_info(
                                f"✓ Decoder '{decoder_name}': Successfully loaded all {loaded_keys} state dict keys from {pretrained_path}"
                            )
                    else:
                        if kwargs.get("verbose", False):
                            _log_info(
                                f"⚠ Decoder '{decoder_name}': Loaded {loaded_keys}/{total_model_keys} state dict keys from {pretrained_path}"
                            )
                        if incompatible_keys:
                            if kwargs.get("verbose", False):
                                _log_info(
                                    f"  - {len(incompatible_keys)} keys had incompatible shapes and were skipped"
                                )
                        if missing_keys:
                            if kwargs.get("verbose", False):
                                _log_info(
                                    f"  - {len(missing_keys)} keys were missing from checkpoint"
                                )
                        if unexpected_keys:
                            if kwargs.get("verbose", False):
                                _log_info(
                                    f"  - {len(unexpected_keys)} unexpected keys in checkpoint"
                                )

                    # Log warnings about incompatible keys
                    if incompatible_keys:
                        import warnings

                        warnings.warn(
                            f"Some weights from {pretrained_path} had incompatible shapes and were skipped:\n"
                            + "\n".join(incompatible_keys[:10])  # Show first 10
                            + (
                                f"\n... and {len(incompatible_keys) - 10} more"
                                if len(incompatible_keys) > 10
                                else ""
                            )
                            + "\nThis is likely because the pretrained model used different feature_dim values."
                        )

                    if missing_keys:
                        import warnings

                        warnings.warn(
                            f"Some keys were missing when loading pretrained decoder from {pretrained_path}: {missing_keys[: min(5, len(missing_keys))]}..."
                        )

                    # Freeze all decoder parameters initially (will be selectively unfrozen below if needed)
                    for param in decoder.parameters():
                        param.requires_grad = False
                    # Don't set to eval mode yet - we may selectively unfreeze below
                    if kwargs.get("verbose", False):
                        _log_info(
                            f"Loaded pre-trained decoder weights from {pretrained_path}"
                        )

                # Freeze decoder if decoder_lr is 0 (or None and from_pretrained was set)
                # If decoder_lr is explicitly 0, freeze regardless of pretrained status
                if decoder_lr is not None and decoder_lr == 0.0:
                    for param in decoder.parameters():
                        param.requires_grad = False
                    decoder.eval()
                    if kwargs.get("verbose", False):
                        _log_info(
                            f"Decoder '{decoder_name}' frozen due to DECODER_LR=0.0"
                        )
                else:
                    # Apply selective freezing if parameters are specified
                    if (
                        num_fusion_blocks_trainable is not None
                        or train_rgb_head is not None
                    ):
                        # If pretrained weights were loaded, we already froze everything above
                        # Otherwise, freeze everything first
                        if not (pretrained_path and pretrained_path != ""):
                            for param in decoder.parameters():
                                param.requires_grad = False

                        # Then selectively unfreeze based on config
                        # Freeze/unfreeze reassemble layers (typically frozen)
                        for param in decoder.reassemble_layers.parameters():
                            param.requires_grad = False

                        # Handle fusion blocks: train the last N blocks (highest indices)
                        # fusion_blocks[3] is smallest scale, fusion_blocks[0] is largest scale
                        if num_fusion_blocks_trainable is not None:
                            num_fusion_blocks_trainable = int(
                                num_fusion_blocks_trainable
                            )
                            num_blocks = len(decoder.fusion_blocks)
                            if (
                                num_fusion_blocks_trainable < 0
                                or num_fusion_blocks_trainable > num_blocks
                            ):
                                raise ValueError(
                                    f"NUM_FUSION_BLOCKS_TRAINABLE must be between 0 and {num_blocks}, "
                                    f"got {num_fusion_blocks_trainable}"
                                )
                            # Train the last N blocks (from the end)
                            # e.g., if num_fusion_blocks_trainable=2, train blocks [2] and [3]
                            start_idx = num_blocks - num_fusion_blocks_trainable
                            for i in range(start_idx, num_blocks):
                                for param in decoder.fusion_blocks[i].parameters():
                                    param.requires_grad = True
                            if kwargs.get("verbose", False):
                                _log_info(
                                    f"Decoder '{decoder_name}': Training {num_fusion_blocks_trainable} "
                                    f"fusion blocks (indices {start_idx} to {num_blocks - 1})"
                                )
                        else:
                            # If not specified, train all fusion blocks
                            for fusion_block in decoder.fusion_blocks:
                                for param in fusion_block.parameters():
                                    param.requires_grad = True

                        # Handle RGB head
                        if train_rgb_head is not None:
                            train_rgb_head = bool(train_rgb_head)
                            # Check if it's ConvNext decoder (has split head stages) or standard decoder
                            if hasattr(decoder, "rgb_head_stage1"):
                                # ConvNext decoder: has rgb_head_stage1, stage2, stage3, final
                                rgb_head_modules = [
                                    decoder.rgb_head_stage1,
                                    decoder.rgb_head_stage2,
                                    decoder.rgb_head_stage3,
                                    decoder.rgb_head_final,
                                ]
                            elif hasattr(decoder, "rgb_head"):
                                # Standard decoder: has single rgb_head Sequential
                                rgb_head_modules = [decoder.rgb_head]
                            else:
                                # FiLM decoder also has rgb_head
                                rgb_head_modules = [decoder.rgb_head]

                            for rgb_head_module in rgb_head_modules:
                                for param in rgb_head_module.parameters():
                                    param.requires_grad = train_rgb_head
                            if kwargs.get("verbose", False):
                                _log_info(
                                    f"Decoder '{decoder_name}': RGB head trainable = {train_rgb_head}"
                                )
                        else:
                            # If not specified, train RGB head by default
                            if hasattr(decoder, "rgb_head_stage1"):
                                for module in [
                                    decoder.rgb_head_stage1,
                                    decoder.rgb_head_stage2,
                                    decoder.rgb_head_stage3,
                                    decoder.rgb_head_final,
                                ]:
                                    for param in module.parameters():
                                        param.requires_grad = True
                            elif hasattr(decoder, "rgb_head"):
                                for param in decoder.rgb_head.parameters():
                                    param.requires_grad = True

                        decoder.train()
                        if kwargs.get("verbose", False):
                            _log_info(
                                f"Decoder '{decoder_name}' selectively frozen/unfrozen"
                            )
                    else:
                        # No selective freezing specified, train everything
                        for param in decoder.parameters():
                            param.requires_grad = True
                        decoder.train()
                        # _log_info(
                        #     f"Decoder '{decoder_name}' un-frozen due to DECODER_LR={decoder_lr}"
                        # )
                # Store decoder_lr for optimizer setup (will be None if not specified)
                decoder.decoder_lr = decoder_lr
                return decoder
            raise TypeError(
                "Decoder must be DPT_Decoder/DPT_Decoder_ConvNext/FiLMConditionedDPT instance or dict."
            )

        self.decoder_names = list(decoders.keys())
        self.decoders = nn.ModuleDict()
        # Store decoder learning rates for optimizer setup
        self.decoder_lrs = {}
        for decoder_name, decoder_config in decoders.items():
            decoder = build_dpt(decoder_config, decoder_name=decoder_name)
            self.decoders[decoder_name] = decoder
            # Store the learning rate for this decoder (None means use default/base LR)
            self.decoder_lrs[decoder_name] = getattr(decoder, "decoder_lr", None)

    def _rgb_tokens(self, rgb_preproc):
        """Extract DINOv3 tokens and infer (Hp, Wp) if wrapper doesn’t return them."""
        with torch.no_grad():
            out = self.dinov3(rgb_preproc)
        tokens = out.get("last_hidden_state", out.get("tokens"))
        if tokens is None:
            raise KeyError(
                "DINOv3 wrapper must return 'last_hidden_state' or 'tokens'."
            )
        Hp = self.image_size // self.patch_size
        Wp = self.image_size // self.patch_size
        return tokens, (Hp, Wp)

    def forward(self, model_input_dict, just_extract_tokens=False):
        if isinstance(model_input_dict, torch.Tensor):
            model_input_dict = {"rgb": model_input_dict}
            model_input_dict["inpaint_mask_dilation"] = 5

        elif isinstance(model_input_dict, dict):
            if "inpaint_mask_dilation" not in model_input_dict:
                model_input_dict["inpaint_mask_dilation"] = 5
        x = model_input_dict["rgb"]  # (B,3,H,W)
        rgb_in = self.dinov3.preprocess_image(x)
        tokens_list = self.dinov3(rgb_in)[
            "selected_hidden_states"
        ]  # List[4] of (B,N,C), PATCH TOKENS ONLY
        if just_extract_tokens:
            return tokens_list
        # For ConvNeXt, ensure features are spatial feature maps [B, C, H, W]
        # For regular DINOv3, they are tokens [B, N, C]
        if self.use_convnext:
            # ConvNeXt outputs might be tokens, convert to spatial if needed
            # But if return_as_feature_maps=True, they should already be spatial
            # Check first feature to determine format
            if len(tokens_list[0].shape) == 3:
                # It's tokens [B, N, C], but ConvNext decoder expects spatial
                # This shouldn't happen if config is correct, but handle gracefully
                _log_warning(
                    "ConvNeXt encoder returned tokens but decoder expects spatial maps. "
                    "Set return_as_feature_maps=True in encoder config."
                )

        # 6) Decode with flexible decoder heads
        outputs = {}
        for decoder_name in self.decoder_names:
            decoder_output = self.decoders[decoder_name](tokens_list)
            outputs[decoder_name] = decoder_output

        # Optional: Add tokens for debugging/analysis
        # outputs.update({
        #     "rgb_tokens": rgb_features,
        # })

        return outputs

    def extract_tokens(self, image):
        rgb_in = self.dinov3.preprocess_image(image)
        tokens_list = self.dinov3(rgb_in)["selected_hidden_states"]
        return tokens_list


class UnReflect_Model_TokenInpainter(UnReflect_Model):
    """
    Assumes decoders dict contains at least:
      - "highlight": DPT_Decoder(output_channels=1) that predicts a soft mask in [0,1]
      - "diffuse":   (regular) DPT_Decoder that expects standard DINO tokens
    Keeps DPT decoders intact; completes tokens first.
    """

    def __init__(
        self,
        dinov3,
        decoders,
        patch_size: int = 16,
        token_inpainter_cfg: dict | None = None,
        detach_inpainted_tokens_for_decoder: bool = True,
        suppress_sky: bool = False,
        sky_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            dinov3=dinov3, decoders=decoders, patch_size=patch_size, **kwargs
        )
        self.detach_inpainted_tokens_for_decoder = bool(
            detach_inpainted_tokens_for_decoder
        )
        # Sky suppression: drop detected sky from the inpaint mask so the token
        # inpainter never repaints the sky (which the highlight head otherwise
        # flags as a giant highlight). A per-forward `suppress_sky` in the input
        # dict overrides this default. `sky_cfg` overrides detect_sky_mask kwargs.
        self.suppress_sky = bool(suppress_sky)
        self.sky_cfg = dict(sky_cfg) if sky_cfg else {}
        dim = self.embed_dim
        # Extract TokenInpainter class and module from config
        if token_inpainter_cfg is None:
            token_inpainter_cfg = {}

        # Get class name and module (with defaults for backward compatibility)
        self.token_inpainter_class_name = token_inpainter_cfg.pop(
            "token_inpainter_class", "TokenInpainter"
        )
        self.token_inpainter_module_name = token_inpainter_cfg.pop(
            "token_inpainter_module", "models"
        )

        # Extract pretrained path before filtering (not passed to constructor)
        pretrained_relative_path = token_inpainter_cfg.pop("from_pretrained", "")
        if pretrained_relative_path != "":
            pretrained_path = os.path.join(
                os.environ.get("WEIGHTS_DIR", "weights"), pretrained_relative_path
            )
        else:
            pretrained_path = None

        # Dynamically import the module
        token_inpainter_module = importlib.import_module(
            self.token_inpainter_module_name
        )

        # Get the TokenInpainter class from the module
        TokenInpainterClass = getattr(
            token_inpainter_module, self.token_inpainter_class_name
        )

        # Filter kwargs to only include parameters that the constructor accepts
        # This prevents errors when passing unexpected parameters
        sig = inspect.signature(TokenInpainterClass.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}  # Exclude 'self'
        filtered_cfg = {
            k: v for k, v in token_inpainter_cfg.items() if k in accepted_params
        }

        # Instantiate the TokenInpainter with filtered config
        self.token_inpaint = TokenInpainterClass(dim=dim, **filtered_cfg)

        # Load pretrained weights if path is specified and not empty
        if pretrained_path and pretrained_path != "":
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(
                    f"Pretrained token inpainter weights not found at: {pretrained_path}"
                )

            # Load checkpoint (handle both raw state_dict and checkpoint formats)
            if not os.path.exists(pretrained_path):
                if pretrained_path.endswith(".pt"):
                    alt_pretrained_path = pretrained_path[:-3] + ".pth"
                elif pretrained_path.endswith(".pth"):
                    alt_pretrained_path = pretrained_path[:-4] + ".pt"
                    alt_pretrained_path = None
                if alt_pretrained_path and os.path.exists(alt_pretrained_path):
                    pretrained_path = alt_pretrained_path
            checkpoint = torch.load(
                pretrained_path, map_location="cpu", weights_only=False
            )

            # Extract state dict (handle both formats)
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                # Try common checkpoint keys
                state_dict = (
                    checkpoint.get("model_state_dict")
                    or checkpoint.get("state_dict")
                    or checkpoint
                )

            # Strip common prefixes if state dict was saved as part of larger model
            # Handle cases like "token_inpaint.weight" -> "weight" or "token_inpainter.weight" -> "weight"
            if isinstance(state_dict, dict) and len(state_dict) > 0:
                sample_key = next(iter(state_dict.keys()))
                if "." in sample_key:
                    # Try to find token inpainter-specific prefix pattern
                    prefix_options = [
                        "module.token_inpaint.",
                        "module.token_inpainter.",
                        "token_inpaint.",
                        "token_inpainter.",
                    ]
                    for prefix in prefix_options:
                        # Check if any keys start with this prefix
                        matching_keys = [
                            k for k in state_dict.keys() if k.startswith(prefix)
                        ]
                        if matching_keys:
                            # Strip token inpainter prefix from matching keys, keep others as-is
                            stripped_dict = {}
                            for k, v in state_dict.items():
                                if k.startswith(prefix):
                                    stripped_dict[k[len(prefix) :]] = v
                                else:
                                    stripped_dict[k] = v
                            state_dict = stripped_dict
                            break

            # Load weights with strict=False to handle partial matches
            # Filter out keys with shape mismatches before loading
            filtered_state_dict = {}
            incompatible_keys = []
            model_state_dict = self.token_inpaint.state_dict()

            for key, value in state_dict.items():
                if key in model_state_dict:
                    model_shape = model_state_dict[key].shape
                    checkpoint_shape = value.shape
                    if model_shape == checkpoint_shape:
                        filtered_state_dict[key] = value
                    else:
                        incompatible_keys.append(
                            f"{key}: checkpoint shape {checkpoint_shape} != model shape {model_shape}"
                        )
                else:
                    # Key not in model, skip it
                    pass

            # Load filtered state dict
            missing_keys, unexpected_keys = self.token_inpaint.load_state_dict(
                filtered_state_dict, strict=False
            )

            # Print confirmation about weight loading status
            total_model_keys = len(model_state_dict)
            loaded_keys = len(filtered_state_dict)
            has_issues = bool(incompatible_keys or missing_keys or unexpected_keys)

            if not has_issues and loaded_keys == total_model_keys:
                if kwargs.get("verbose", False):
                    _log_info(
                        f"✓ Token Inpainter: Successfully loaded all {loaded_keys} state dict keys from {pretrained_path}"
                    )
            else:
                if kwargs.get("verbose", False):
                    _log_info(
                        f"⚠ Token Inpainter: Loaded {loaded_keys}/{total_model_keys} state dict keys from {pretrained_path}"
                    )
                if incompatible_keys:
                    if kwargs.get("verbose", False):
                        _log_info(
                            f"  - {len(incompatible_keys)} keys had incompatible shapes and were skipped"
                        )
                if missing_keys:
                    if kwargs.get("verbose", False):
                        _log_info(
                            f"  - {len(missing_keys)} keys were missing from checkpoint"
                        )
                if unexpected_keys:
                    if kwargs.get("verbose", False):
                        _log_info(
                            f"  - {len(unexpected_keys)} unexpected keys in checkpoint"
                        )

            # Log warnings about incompatible keys
            if incompatible_keys:
                import warnings

                warnings.warn(
                    f"Some weights from {pretrained_path} had incompatible shapes and were skipped:\n"
                    + "\n".join(incompatible_keys[:10])  # Show first 10
                    + (
                        f"\n... and {len(incompatible_keys) - 10} more"
                        if len(incompatible_keys) > 10
                        else ""
                    )
                    + "\nThis is likely because the pretrained model used different architecture parameters."
                )

            if missing_keys:
                if kwargs.get("verbose", False):
                    _log_warning(
                        f"Missing keys when loading pretrained token inpainter from {pretrained_path}: {missing_keys[:5]}"
                        + (
                            f" (and {len(missing_keys) - 5} more)"
                            if len(missing_keys) > 5
                            else ""
                        )
                    )

            if unexpected_keys:
                if kwargs.get("verbose", False):
                    _log_warning(
                        f"Unexpected keys when loading pretrained token inpainter from {pretrained_path}: {unexpected_keys[:5]}"
                        + (
                            f" (and {len(unexpected_keys) - 5} more)"
                            if len(unexpected_keys) > 5
                            else ""
                        )
                    )

            if kwargs.get("verbose", False):
                _log_info(
                    f"Loaded pretrained token inpainter weights from {pretrained_path}"
                )

    def extract_tokens(self, image):
        rgb_in = self.dinov3.preprocess_image(image)
        tokens_list = self.dinov3(rgb_in)["selected_hidden_states"]
        return tokens_list

    def forward(
        self,
        model_input_dict,
        threshold=0.3,
        dilation=None,
        just_extract_tokens=False,
    ):
        if isinstance(
            model_input_dict, torch.Tensor
        ):  # Tensor passed directly, we put it into "rgb" key
            model_input_dict = {"rgb": model_input_dict}
            model_input_dict["inpaint_mask_dilation"] = (
                model_input_dict["rgb"].shape[-1] // (self.patch_size * 4)
                if dilation in [None, -1]
                else dilation
            )
            model_input_dict["inpaint_mask_threshold"] = threshold
        elif isinstance(model_input_dict, dict):  # Passed a dict as input
            # inpaint_mask_dilation: None/-1 → auto from rgb size; else dict wins over arg
            if dilation in (None, -1):
                model_input_dict["inpaint_mask_dilation"] = (
                    model_input_dict["rgb"].shape[-1] // (self.patch_size * 4)
                )
            else:
                model_input_dict["inpaint_mask_dilation"] = model_input_dict.get(
                    "inpaint_mask_dilation", dilation
                )
            if "inpaint_mask_threshold" not in model_input_dict:
                model_input_dict["inpaint_mask_threshold"] = threshold
        model_input_dict["inpaint_mask_dilation"] = int(
            max(1, model_input_dict["inpaint_mask_dilation"])
        )
        x = model_input_dict["rgb"]  # (B,3,H,W)
        rgb_in = self.dinov3.preprocess_image(x)
        tokens_list = self.dinov3(rgb_in)[
            "selected_hidden_states"
        ]  # List[4] of (B,N,C), PATCH TOKENS ONLY
        if just_extract_tokens:
            return tokens_list
        outputs = {}

        ### FIRST: Predict soft highlight mask in image space
        if "highlight" not in self.decoders:
            raise KeyError("decoders must include a 'highlight' head for Option B.")
        outputs["highlight"] = self.decoders["highlight"](
            tokens_list
        )  # (B,1,H,W) in [0,1]
        # hl_soft = self.decoders["highlight"](tokens_list) # (B,1,H,W) in [0,1]

        ### SECOND: Construct patch-level mask - From the prediction or override from GT if provided
        # Get the initial mask (GT override if present, else from highlight prediction)
        pixel_inpaint_mask = model_input_dict.get(
            "inpaint_mask_override", outputs["highlight"]
        )

        model_input_dict["inpaint_mask_dilation"] = model_input_dict["inpaint_mask_dilation"]+1 if model_input_dict["inpaint_mask_dilation"] % 2 == 0 else model_input_dict["inpaint_mask_dilation"]
        pixel_inpaint_mask = (
            torch.nn.functional.max_pool2d(
                pixel_inpaint_mask.float(),
                kernel_size=model_input_dict["inpaint_mask_dilation"],
                stride=1,
                padding=model_input_dict["inpaint_mask_dilation"] // 2,
            )
            > model_input_dict["inpaint_mask_threshold"]
        )

        ### Sky suppression: never inpaint detected sky. Removes sky from the
        # inpaint mask so the token inpainter leaves it untouched (the highlight
        # head tends to flag bright sky as one large highlight).
        suppress_sky = bool(
            model_input_dict.get("suppress_sky", getattr(self, "suppress_sky", False))
        )
        if suppress_sky:
            sky_cfg = model_input_dict.get("sky_cfg", getattr(self, "sky_cfg", {})) or {}
            sky_mask = detect_sky_mask(x, **sky_cfg)  # (B,1,H,W) float, x in [0,1]
            if sky_mask.shape[-2:] != pixel_inpaint_mask.shape[-2:]:
                sky_mask = F.interpolate(
                    sky_mask, size=pixel_inpaint_mask.shape[-2:], mode="nearest"
                )
            outputs["sky_mask"] = sky_mask
            pixel_inpaint_mask = pixel_inpaint_mask & (sky_mask <= 0.5)

        outputs["highlight_mask"] = pixel_inpaint_mask.float()
        patch_inpaint_mask = pixel_mask_to_patch_mask(
            pixel_inpaint_mask,
            patch_size=self.patch_size,
            threshold=0,
            invert=False,
            soft="soft" in self.token_inpainter_class_name.lower(),
        )

        if "soft" in self.token_inpainter_class_name.lower():
            patch_inpaint_mask = feather_token_mask(
                patch_inpaint_mask, radius_tokens=1, smoothstep=True
            )
            
        patch_grid_size = int(patch_inpaint_mask.shape[1] ** 0.5)
        outputs["patch_mask"] = patch_inpaint_mask.reshape(
            -1, 1, patch_grid_size, patch_grid_size
        ).float()
        # 1 if the patch needs to be inpainted, 0 if not
        # outputs["patch_mask"] = pixel_inpaint_mask.int()

        ### THIRD: Inpaint the tokens in the mask
        # Detect if using soft masks (float) or boolean masks
        is_soft_mask = patch_inpaint_mask.dtype.is_floating_point

        completed_tokens = []  # With gradients - for token loss
        completed_tokens_detached = []  # Detached - for decoders (prevents decoder loss from affecting TokenInpainter)
        # Token-inpainter mask polarity. The TokenInpainter_Prior contract is
        # True/1.0 = hole-to-inpaint (it overwrites True positions with the learned
        # mask_token + clean-neighbour prior). The released checkpoint, however, was
        # trained feeding the *visibility* mask (True = visible/teacher, False = hole) —
        # i.e. with the polarity INVERTED vs the contract — so by default we keep that
        # inversion for checkpoint compatibility. Set INPAINT_FEED_RAW_MASK=True
        # (passed in as 'inpaint_feed_raw_mask') ONLY for a fresh-from-scratch run to
        # feed the design-correct raw mask; this is INCOMPATIBLE with the released weights.
        feed_raw_mask = bool(model_input_dict.get("inpaint_feed_raw_mask", False))
        for n, T in enumerate(tokens_list):  # (B,N,C)
            if feed_raw_mask:
                # Design-correct: True/1.0 = hole/highlight to inpaint.
                inpaint_arg = patch_inpaint_mask
            elif is_soft_mask:
                inpaint_arg = 1.0 - patch_inpaint_mask  # [B, N] float in [0,1]
            else:
                inpaint_arg = torch.logical_not(patch_inpaint_mask)  # [B, N] bool
            T_inpainted = self.token_inpaint(T, inpaint_arg)

            # Blend: keep teacher tokens on context; use predicted tokens on masked patches
            if is_soft_mask:
                # Soft blending: patch_inpaint_mask=1.0 → use T_inpainted, patch_inpaint_mask=0.0 → use T
                patch_inpaint_mask_expanded = patch_inpaint_mask.unsqueeze(
                    -1
                )  # [B, N, 1]
                T_comp = (
                    patch_inpaint_mask_expanded * T_inpainted
                    + (1.0 - patch_inpaint_mask_expanded) * T
                )  # (B,N,C)
            else:
                # Putting the inpainted tokens where the patch_inpaint_mask is 1
                T_comp = torch.where(
                    patch_inpaint_mask.unsqueeze(-1), T_inpainted, T
                )  # (B,N,C)

            # Store with gradients for token loss
            completed_tokens.append(T_comp)
            # Store detached version for decoders (prevents decoder loss gradients from reaching TokenInpainter)
            completed_tokens_detached.append(T_comp.detach())

        # Store tokens_completed WITH gradients for token loss computation
        outputs["tokens_completed"] = completed_tokens
        outputs["tokens_inpainted"] = T_inpainted
        outputs["tokens_input"] = tokens_list

        # 4) Decode. By default the inpainter is shielded from decoder gradients
        # (set DETACH_INPAINTED_TOKENS_FOR_DECODER: False at end2end to let decoder
        # losses correct token-level residuals visible as colour artifacts).
        decoder_input_tokens = (
            completed_tokens_detached
            if self.detach_inpainted_tokens_for_decoder
            else completed_tokens
        )
        for name, dec in self.decoders.items():
            if name == "highlight":
                continue
            outputs[name] = dec(decoder_input_tokens)

        return outputs
