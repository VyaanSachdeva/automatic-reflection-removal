"""Download assets API for UnReflectAnything."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

from os import PathLike

from ._shared import (
    download_configs,
    download_images,
    download_notebooks,
    download_weights,
    get_cache_dir,
)


def download(
    what: Optional[
        Union[
            Literal["weights", "images", "notebooks", "configs", "all"],
            list[Literal["weights", "images", "notebooks", "configs"]]
        ]
    ] = None,
    *,
    asset: Optional[
        Union[
            Literal["weights", "images", "notebooks", "configs", "all"],
            list[Literal["weights", "images", "notebooks", "configs"]]
        ]
    ] = None,
    output_dir: Optional[Union[str, PathLike, Path]] = None,
    variant: str = "default",
    force: bool = False,
) -> Union[Path, dict[str, Path]]:
    """Download assets from the HuggingFace repository.

    Same behavior as the CLI ``unreflectanything download --weights`` / ``--images`` /
    ``--notebooks`` / ``--all``.

    Args:
        what: What to download (positional), or list thereof. Use ``asset=`` as alias.
        asset: Alias for ``what``.
        output_dir: Directory to save downloaded files. If None, uses the
            default cache directory (~/.cache/unreflectanything/).
        variant: Weights variant to download (e.g., "default").
        force: If True, re-download even if files already exist.

    Returns:
        Path to the directory where files were saved, or a dict of paths for multiple requested assets.
    """
    resolved_what = asset if asset is not None else what
    valid_assets = ("weights", "images", "notebooks", "configs")
    valid_single = valid_assets + ("all",)

    # Handle no input: default to "weights"
    if resolved_what is None:
        resolved_what = "weights"

    # Expand "all"
    if resolved_what == "all":
        asset_list = list(valid_assets)
    elif isinstance(resolved_what, (list, tuple)):
        asset_list = []
        for elem in resolved_what:
            if elem == "all":
                asset_list.extend(valid_assets)
            elif elem in valid_assets:
                asset_list.append(elem)
            else:
                raise ValueError(
                    f"Invalid value in asset list: {elem!r}. "
                    f"Must be one of {valid_assets} or 'all'."
                )
        # remove duplicates, preserve order
        seen = set()
        asset_list = [a for a in asset_list if not (a in seen or seen.add(a))]
    elif resolved_what in valid_single:
        if resolved_what in valid_assets:
            asset_list = [resolved_what]
        else:  # "all" already handled
            asset_list = list(valid_assets)
    else:
        raise ValueError(
            f"Invalid 'what'/'asset' value: {resolved_what!r}. "
            "Must be 'weights', 'images', 'notebooks', 'configs', 'all', or a list thereof."
        )

    # Output path setup
    if output_dir is None:
        output_path = get_cache_dir("weights").parent
    else:
        output_path = Path(output_dir).expanduser().resolve()

    output_path.mkdir(parents=True, exist_ok=True)

    # Download the requested assets
    asset_dirs = {}
    for asset_item in asset_list:
        if asset_item == "weights":
            weights_dir = output_path / "weights"
            WEIGHTS_REVISION="a0d3c7bff5ddb2c430e74b5f8ee67be7cc28fcbe"
            download_weights(output_dir=weights_dir, variant=WEIGHTS_REVISION, force=force)
            asset_dirs["weights"] = weights_dir
        elif asset_item == "images":
            images_dir = output_path / "images"
            download_images(output_dir=images_dir, force=force)
            asset_dirs["images"] = images_dir
        elif asset_item == "notebooks":
            notebooks_dir = output_path / "notebooks"
            download_notebooks(output_dir=notebooks_dir, force=force)
            asset_dirs["notebooks"] = notebooks_dir
        elif asset_item == "configs":
            configs_dir = output_path / "configs"
            download_configs(output_dir=configs_dir, force=force)
            asset_dirs["configs"] = configs_dir
        else:
            # Defensive: Should not be possible at this point
            raise RuntimeError(f"Unknown asset type: {asset_item!r}")

    # Return: single asset returns the Path, multiple returns dict
    if len(asset_dirs) == 1:
        return next(iter(asset_dirs.values()))
    else:
        return asset_dirs
