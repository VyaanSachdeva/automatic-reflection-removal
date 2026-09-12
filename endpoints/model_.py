"""Model factory and UnReflectModel wrapper for UnReflectAnything.

Provides two public symbols:

* :func:`model` – thin convenience factory (delegates to :class:`UnReflectModel`).
* :class:`UnReflectModel` – ``nn.Module`` wrapper for tensor-in / tensor-out
  inference **and** weight-free instantiation for training.

Resolution order (applies to both config and weights):

1. In-memory object (``config`` / ``weights``)  – used directly.
2. Filesystem path   (``config_path`` / ``weights_path``) – loaded from disk.
3. Default cached path in ``~/.cache/unreflectanything/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union, Any

from os import PathLike

if TYPE_CHECKING:
    from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nn_module_base():
    """Lazy-import ``nn.Module`` so ``torch`` is not loaded at module level."""
    import torch.nn as nn

    return nn.Module


def _is_path(x) -> bool:
    """Return True when *x* looks like a filesystem path (str / Path / os.PathLike)."""
    return isinstance(x, (str, Path)) or getattr(x, "__fspath__", None) is not None


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def model(
    pretrained: bool = True,
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    device: str = "cuda",
    config_path: Optional[Union[str, PathLike, Path]] = None,
    verbose: bool = False,
    config: Optional[Union[dict, Any]] = None,
    weights: Optional[Union[str, PathLike, Path, dict, Any]] = None,
    strict: bool = True,
) -> "UnReflectModel":
    """Convenience factory — delegates entirely to :class:`UnReflectModel`.

    See :meth:`UnReflectModel.__init__` for full parameter documentation.

    Returns:
        An ``UnReflectModel`` instance (with or without pretrained weights).
    """
    return UnReflectModel(
        pretrained=pretrained,
        weights_path=weights_path,
        device=device,
        config_path=config_path,
        verbose=verbose,
        config=config,
        weights=weights,
        strict=strict,
    )


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class UnReflectModel(_nn_module_base()):
    """``nn.Module`` wrapper around the inner UnReflect model.

    The inner model is stored as the ``_model`` submodule so that ``.to()``,
    ``.eval()``, ``.parameters()``, and ``state_dict()`` propagate correctly.

    Attributes:
        image_size (int): Spatial size expected by the encoder (e.g. 448).
        device (torch.device): Device the model lives on (read-only property).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[Union[str, PathLike, Path]] = None,
        device: str = "cuda",
        config_path: Optional[Union[str, PathLike, Path]] = None,
        verbose: bool = False,
        config: Optional[Union[dict, Any]] = None,
        weights: Optional[Union[str, PathLike, Path, dict, Any]] = None,
        strict: bool = True,
    ):
        """Build (and optionally load weights for) the UnReflect model.

        Args:
            pretrained: When ``True`` (default) load checkpoint weights after
                building the architecture.  When ``False`` only build the
                architecture — useful for custom training or fine-tuning.
            weights_path: Path to a ``.pt`` checkpoint file.  Falls back to the
                cached ``full_model_weights.pt`` when ``None``.
                *Ignored when* ``weights`` *is provided.*
            device: Target device string (``"cuda"``, ``"cpu"``, ``"cuda:1"``, …).
            config_path: Path to a YAML config describing the architecture.
                Falls back to the cached ``pretrained_config.yaml`` when ``None``.
                *Ignored when* ``config`` *is provided.*
            verbose: Print progress messages.
            config: In-memory configuration (dict or DotMap).
                **Overrides** ``config_path`` when provided.
            weights: In-memory state-dict **or** path to a checkpoint.
                **Overrides** ``weights_path`` when provided.
        """
        super().__init__()
        import torch
        from ._shared import _resolve_device
        from ._model_builder import create_model_from_config_minimal
        # ── 1. Resolve target device ──────────────────────────────────
        torch_device = torch.device(_resolve_device(device))

        # ── 2. Load checkpoint up-front ───────────────────────────────
        #    Loaded *before* building so the checkpoint's embedded training
        #    config can drive the architecture (see `_resolve_config`). This
        #    guarantees the built model matches the weights instead of relying
        #    on the shipped YAML staying in sync with the released checkpoint —
        #    a mismatch in non-parametric forward hyper-parameters (e.g. the
        #    token-inpainter local prior) loads cleanly under strict=True yet
        #    silently changes the output.
        checkpoint = (
            self._load_checkpoint(weights, weights_path, torch_device, verbose)
            if pretrained
            else None
        )

        # ── 3. Resolve model configuration ────────────────────────────
        #    Priority: explicit `config` / `config_path`  >  config embedded in
        #    the checkpoint (sanitized for inference)  >  packaged
        #    pretrained_config.yaml.
        model_config = self._resolve_config(config, config_path, checkpoint, verbose)

        # ── 4. Build model architecture (no utilities/engine imports) ──
        inner = create_model_from_config_minimal(model_config, torch_device, verbose=verbose)

        # ── 5. Load pretrained weights (skipped when pretrained=False) ─
        if pretrained:
            sd = self._extract_state_dict(checkpoint)
            sd = self._strip_key_prefixes(inner, sd, verbose)
            inner.load_state_dict(sd, strict=strict)
            inner.eval()
            if verbose:
                print("Pretrained weights loaded; model set to eval mode.")

        # ── 5. Register submodule & cache metadata ────────────────────
        self._model = inner
        self._device = torch_device

        # Read the spatial size the encoder expects (e.g. 448)
        encoder_cfg = self._model.dinov3.config
        self.image_size = getattr(
            encoder_cfg,
            "image_size",
            encoder_cfg.get("image_size", 448)
            if hasattr(encoder_cfg, "get")
            else 448,
        )

    # ------------------------------------------------------------------
    # Internal resolution helpers (static — no instance state needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config(config, config_path, checkpoint, verbose):
        """Return a parsed DotMap config describing the architecture to build.

        Resolution priority:
            1. Explicit in-memory ``config`` (user-supplied) — used as-is.
            2. Explicit ``config_path`` (user-supplied YAML) — loaded from disk.
            3. Config embedded in ``checkpoint`` (``checkpoint["config"]``),
               sanitized for inference. Following the checkpoint's own config
               keeps the architecture in lock-step with the weights, so a stale
               shipped YAML can no longer silently corrupt the model.
            4. Packaged ``pretrained_config.yaml`` (fallback when there is no
               embedded config — e.g. a bare state-dict or ``pretrained=False``).
        """
        from ._shared import DEFAULT_CONFIG_FILENAME
        from ._model_builder import load_config_minimal

        # 1. In-memory config — highest priority; respect explicit user input.
        if config is not None:
            parsed = load_config_minimal(config=config)
            if parsed is None:
                raise ValueError(
                    "config could not be parsed (expected dict or DotMap)."
                )
            if verbose:
                print("Loaded model configuration from in-memory object.")
            return parsed

        # 2. Explicit config file.
        if config_path is not None:
            config_path = Path(config_path)
            if config_path.is_dir():
                config_path = config_path / DEFAULT_CONFIG_FILENAME
            parsed = load_config_minimal(config_path=config_path)
            if verbose:
                print(f"Loaded model configuration from: `{config_path}`")
            return parsed

        # 3. Config embedded in the checkpoint (sanitized for inference).
        embedded = UnReflectModel._extract_embedded_config(checkpoint)
        if embedded is not None:
            sanitized = UnReflectModel._sanitize_embedded_config(embedded, verbose)
            if verbose:
                print("Built architecture from the checkpoint's embedded training config.")
            return load_config_minimal(config=sanitized)

        # 4. Packaged pretrained_config.yaml (fallback).
        packaged = UnReflectModel._packaged_config_path()
        parsed = load_config_minimal(config_path=packaged)
        if verbose:
            print(f"Loaded model configuration from packaged default: `{packaged}`")
        return parsed

    @staticmethod
    def _packaged_config_path():
        """Path to the ``pretrained_config.yaml`` shipped inside the package."""
        import importlib.resources

        from ._shared import DEFAULT_CONFIG_FILENAME

        pkg = importlib.resources.files("unreflectanything")
        return pkg / "assets" / DEFAULT_CONFIG_FILENAME

    @staticmethod
    def _packaged_encoder_name():
        """Encoder name declared in the packaged config (a public mirror), or None."""
        from ._model_builder import load_config_minimal

        try:
            cfg = load_config_minimal(config_path=UnReflectModel._packaged_config_path())
            enc = cfg.MODEL.RGB_ENCODER.ENCODER
            return enc if isinstance(enc, str) and enc else None
        except Exception:
            return None

    @staticmethod
    def _extract_embedded_config(checkpoint):
        """Return the architecture config embedded in a training checkpoint, or None.

        A released checkpoint is a dict like
        ``{"model_state_dict": ..., "config": <DotMap>, ...}`` whose ``config``
        is the authoritative description of the architecture the weights belong
        to. A bare state-dict has no such key.
        """
        if not isinstance(checkpoint, dict):
            return None
        cfg = checkpoint.get("config")
        if cfg is None:
            return None
        # Guard against a stray "config" tensor key: a real config has a MODEL section.
        try:
            return cfg if "MODEL" in cfg else None
        except TypeError:
            return None

    @staticmethod
    def _sanitize_embedded_config(embedded, verbose):
        """Return an inference-safe plain-dict copy of a checkpoint's training config.

        Two adjustments make a *training* config safe to build from at inference:

        * **Encoder** — training used the gated upstream DINOv3 repo; substitute
          the encoder name the packaged config ships (a public mirror) so users
          without gated access can build the (architecturally identical) model.
          The encoder *weights* come from the checkpoint regardless.
        * **FROM_PRETRAINED** — training seeded sub-modules from local decoder /
          token-inpainter files that don't exist on an end-user machine (and
          would raise ``FileNotFoundError`` during construction). The full
          checkpoint state-dict supersedes them, so these references are dropped.
        """
        import copy

        cfg = embedded.toDict() if hasattr(embedded, "toDict") else copy.deepcopy(dict(embedded))
        model = cfg.get("MODEL")
        if not isinstance(model, dict):
            return cfg

        # Encoder: prefer the packaged (public-mirror) encoder name.
        rgb = model.get("RGB_ENCODER")
        if isinstance(rgb, dict):
            mirror = UnReflectModel._packaged_encoder_name()
            if mirror and rgb.get("ENCODER") != mirror:
                if verbose:
                    print(
                        f"Encoder '{rgb.get('ENCODER')}' -> '{mirror}' "
                        "(packaged public mirror; weights come from the checkpoint)."
                    )
                rgb["ENCODER"] = mirror

        # Strip FROM_PRETRAINED sub-weight references wherever they appear.
        def _strip(d):
            if isinstance(d, dict):
                for key in ("FROM_PRETRAINED", "from_pretrained"):
                    if d.get(key):
                        d[key] = ""

        decoders = model.get("DECODERS")
        if isinstance(decoders, dict):
            for params in decoders.values():
                _strip(params)
        _strip(model.get("DECODER"))
        _strip(model.get("TOKEN_INPAINTER"))
        return cfg

    @staticmethod
    def _load_checkpoint(weights, weights_path, device, verbose):
        """Load and return the raw checkpoint object (full dict or bare state-dict).

        Source resolution order:
            1. ``weights`` (in-memory state-dict **or** path) — overrides ``weights_path``.
            2. ``weights_path`` (explicit path).
            3. Default cached checkpoint file.

        The full object is returned *without* unwrapping ``model_state_dict`` so
        the caller can also read an embedded ``config``. Use
        :meth:`_extract_state_dict` to obtain the flat state-dict for loading.
        """
        import torch
        from ._shared import DEFAULT_WEIGHTS_FILENAME, get_cache_dir

        # `weights` overrides `weights_path`; default to the cached checkpoint.
        source = weights if weights is not None else weights_path
        if source is None:
            source = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME

        # In-memory state-dict (already the weights themselves).
        if not _is_path(source):
            if verbose:
                print("Using in-memory state dict.")
            return source

        # Filesystem path.
        resolved = Path(source).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Weights file not found at '{resolved}'.\n"
                "Run 'unreflectanything download --weights' or "
                "unreflectanything.download('weights') first."
            )
        if verbose:
            print(f"Loading weights from: `{resolved}`")
        return torch.load(resolved, map_location=device, weights_only=False)

    @staticmethod
    def _extract_state_dict(checkpoint):
        """Return the flat parameter state-dict from a checkpoint object.

        Unwraps the common training-checkpoint wrapper
        (``{"model_state_dict": {...}, ...}``); a bare state-dict is returned
        unchanged.
        """
        if isinstance(checkpoint, dict):
            return checkpoint.get("model_state_dict", checkpoint)
        return checkpoint

    @staticmethod
    def _strip_key_prefixes(model_module, state_dict, verbose):
        """Strip DDP / wrapper prefixes from checkpoint keys when they don't
        match the model's own parameter names.

        Handles common prefixes: ``module.``, ``model.``, ``_model.``.
        """
        if not isinstance(state_dict, dict) or not state_dict:
            return state_dict

        model_keys = set(model_module.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        # If at least some keys already match, no stripping needed
        if model_keys & ckpt_keys:
            return state_dict

        for prefix in ("module.", "model.", "_model."):
            if all(k.startswith(prefix) for k in ckpt_keys):
                state_dict = {
                    k.removeprefix(prefix): v for k, v in state_dict.items()
                }
                if verbose:
                    print(f"Stripped '{prefix}' prefix from state-dict keys.")
                break
        return state_dict

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self):
        """``torch.device`` the model lives on."""
        return self._device

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        images: "Tensor",
        threshold: float = 0.3,
        dilation = 40,
        inpaint_mask_override: Optional["Tensor"] = None,
        return_dict: bool = False,
        composite: bool = False,
    ) -> Union["Tensor", Dict[str, "Tensor"]]:
        """Run inference on a batch of RGB images.

        Args:
            images: ``[B, 3, H, W]`` float tensor with values in ``[0, 1]``.
            inpaint_mask_override: Optional ``[B, 1, H, W]`` binary mask to
                force specific inpainting regions.
            return_dict: If ``True`` return the full output dict; otherwise
                return only the diffuse tensor.
            composite: If ``True``, blend the predicted diffuse with the input
                RGB at non-mask pixels using the model's predicted highlight
                mask. Guarantees pixel-perfect identity outside the mask.

        Returns:
            ``[B, 3, H, W]`` diffuse tensor (clamped to [0, 1]), or the full
            output dict when ``return_dict=True``.
        """
        import torch
        import torch.nn.functional as F

        # Validate input shape
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(
                f"images must be [B, 3, H, W], got shape {tuple(images.shape)}"
            )

        # Build the batch dict expected by the inner model
        rgb_in = images.to(device=self._device, dtype=torch.float32)  # [B,3,H,W]
        batch = {
            "rgb": rgb_in,
            "inpaint_mask_threshold": threshold,
            "inpaint_mask_dilation": dilation,
        }
        if inpaint_mask_override is not None:
            batch["inpaint_mask_override"] = inpaint_mask_override.to(
                device=self._device, dtype=torch.float32,
            )  # [B,1,H,W]

        # Inference (always in eval / no-grad)
        self._model.eval()
        with torch.no_grad():
            out = self._model(batch)

        # Extract and clamp the diffuse component
        diffuse = out.get("diffuse")  # [B,3,H,W]
        highlight = out.get("highlight")  # [B,1,H,W]
        diffuse = diffuse.clamp(0.0, 1.0)  # [B,3,H,W]

        if composite:
            m = out.get("highlight_mask")  # [B,1,H,W] dilated pixel mask
            if m is not None:
                m = F.avg_pool2d(m.float(), kernel_size=5, stride=1, padding=2)
                m = m.clamp(0.0, 1.0)
                if m.shape[-2:] != diffuse.shape[-2:]:
                    m = F.interpolate(
                        m, size=diffuse.shape[-2:], mode="bilinear", align_corners=False
                    )
                diffuse = diffuse * m + rgb_in * (1.0 - m)
                diffuse = diffuse.clamp(0.0, 1.0)

        if return_dict:
            out["diffuse"] = diffuse
            out["highlight"] = highlight
            return out
        return diffuse

    # ------------------------------------------------------------------
    # Train / eval mode helpers
    # ------------------------------------------------------------------

    def eval(self):
        """Set the inner model to eval mode."""
        self._model.eval()
        return self

    def train(self, mode: bool = True):
        """Set the inner model to train mode (for fine-tuning)."""
        self._model.train(mode)
        return self
