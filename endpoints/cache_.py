"""Cache directory management for UnReflectAnything.

Provides helpers that resolve the platform-specific cache root used for
weights, images, notebooks, and config files.

Python API::

    from unreflectanything import cache, cache_clear

    base   = cache()                  # ~/.cache/unreflectanything
    wdir   = cache("weights")         # ~/.cache/unreflectanything/weights

    cache_clear("weights")            # delete weights subdir
    cache_clear()                     # delete entire cache

CLI::

    unreflectanything cache --dir
    unreflectanything cache --dir --weights
    unreflectanything cache --clear --weights
    unreflectanything cache --clear            # everything
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

_VALID_SUBDIRS = ("weights", "images", "notebooks", "configs", "")


def get_cache_dir(subdir: Optional[str] = "") -> Path:
    """Return the default base directory for caching downloaded assets (cross-platform).

    - **Linux / macOS**: Uses ``$XDG_CACHE_HOME`` if set (XDG Base Dir spec), otherwise
      ``~/.cache``. Result: ``$XDG_CACHE_HOME/unreflectanything`` or
      ``~/.cache/unreflectanything``.
    - **Windows**: Uses ``%LOCALAPPDATA%`` if set (e.g. ``C:\\Users\\...\\AppData\\Local``),
      otherwise ``~/.cache`` (``~`` expands to the user profile). Result:
      ``%LOCALAPPDATA%\\unreflectanything`` or ``~/.cache/unreflectanything``.

    Args:
        subdir: Optional asset subdirectory. One of ``"weights"``, ``"images"``,
            ``"notebooks"``, ``"configs"``, or ``""`` (base dir).

    Returns:
        Resolved :class:`~pathlib.Path` to the (sub)cache directory.
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~/.cache"))
    else:
        base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))

    if subdir not in _VALID_SUBDIRS:
        import warnings

        warnings.warn(
            f"Unknown asset subdir '{subdir}', returning parent cache dir. "
            f"Valid options: {_VALID_SUBDIRS[:-1]}."
        )
        subdir = ""

    return Path(base).expanduser().resolve() / "unreflectanything" / subdir


def cache(asset_name: Optional[str] = None) -> Path:
    """Return the cache directory for UnReflectAnything assets.

    Called with no argument (or ``None``) returns the base cache directory;
    passing an asset name returns the path for that asset subdir.

    Args:
        asset_name: ``"weights"`` | ``"images"`` | ``"notebooks"`` | ``"configs"``
            | ``None``.  ``None`` returns the base directory.

    Returns:
        Resolved :class:`~pathlib.Path`.
    """
    return get_cache_dir(asset_name if asset_name is not None else "")


def cache_dir(subdir: Optional[str] = None) -> Path:
    """Return the cache directory (alias for :func:`cache`). Deprecated: use ``cache()``."""
    return cache(subdir)


def cache_clear(asset_name: Optional[str] = None) -> Path:
    """Delete the cache directory (or a specific asset subdirectory).

    Args:
        asset_name: ``"weights"`` | ``"images"`` | ``"notebooks"`` | ``"configs"``
            to delete only that subdirectory, or ``None`` to delete the entire
            ``unreflectanything`` cache tree.

    Returns:
        The :class:`~pathlib.Path` that was removed (even if it did not exist).
    """
    target = cache(asset_name)
    if target.exists():
        shutil.rmtree(target)
    print(f"Cleared {target}")
