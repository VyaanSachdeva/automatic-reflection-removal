"""
Where this project's data lives: the raindrop_project folder (datasets, demo
assets, DeRaindrop weights, checkpoints). By default it sits next to this repo,
e.g. Downloads/UnReflectAnything and Downloads/raindrop_project, so the code
runs unchanged on any machine with that layout. Set RAINDROP_ROOT to put it
somewhere else.
"""
import os
from pathlib import Path

RAINDROP_ROOT = Path(os.environ.get("RAINDROP_ROOT", Path(__file__).resolve().parent.parent / "raindrop_project"))
