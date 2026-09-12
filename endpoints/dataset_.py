"""Image directory dataset for UnReflectAnything API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

try:
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision.transforms import functional as TF
except ImportError:
    Dataset = None  # type: ignore[misc, assignment]
    Image = None  # type: ignore[misc, assignment]
    TF = None  # type: ignore[misc, assignment]

from ._shared import DEFAULT_IMAGE_EXTENSIONS, _collect_image_paths


if Dataset is not None:

    class ImageDirDataset(Dataset):  # type: ignore[no-redef]
        """
        Dataset that reads images from a directory and returns tensors.

        Each item is a tensor of shape (3, H, W) in [0, 1], optionally resized.
        """

        def __init__(
            self,
            root_dir: Union[str, Path],
            extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
            target_size: Optional[Tuple[int, int]] = None,
            return_path: bool = False,
        ):
            """
            Args:
                root_dir: Directory to scan for images (recursive).
                extensions: File suffixes to consider (e.g. (".png", ".jpg")).
                target_size: If set, (H, W) to resize each image; antialias used.
                return_path: If True, __getitem__ returns (tensor, path_str).
            """
            self.root = Path(root_dir)
            self.extensions = tuple(extensions)
            self.target_size = target_size  # (H, W) or None
            self.return_path = return_path
            self.paths = _collect_image_paths(self.root, self.extensions)
            if not self.paths:
                raise FileNotFoundError(f"No images found under {self.root}")

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int):
            path = self.paths[idx]
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                x = TF.to_tensor(rgb)  # (3, H, W), float32, [0, 1]
            if self.target_size is not None:
                x = TF.resize(x, self.target_size, antialias=True)  # (3, H_t, W_t)
            if self.return_path:
                return x, str(path)
            return x
else:
    ImageDirDataset = None  # type: ignore[misc, assignment]
