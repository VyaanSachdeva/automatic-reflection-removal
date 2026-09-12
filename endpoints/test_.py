"""Test/evaluation pipeline API for UnReflectAnything."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

from os import PathLike


def test(
    config: Union[str, PathLike, Path] = "config/test.yaml",
    **overrides,
) -> None:
    """Run the test/evaluation pipeline.

    This function evaluates a trained UnReflectAnything model using the
    specified configuration. The model checkpoint is determined by the
    RUN parameter in the config.

    Args:
        config: Path to the test configuration YAML file.
        **overrides: Additional config overrides in the format PARAM=value.
            These override values from the config file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        RuntimeError: If testing fails or RUN is not specified.
    """
    config_path = Path(config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    argv_backup = sys.argv
    new_argv = [sys.argv[0], "--config", str(config_path)]

    for key, value in overrides.items():
        new_argv.append(f"--{key.upper()}={value}")

    try:
        sys.argv = new_argv
        import main

        main.run_pipeline(mode="test")
    finally:
        sys.argv = argv_backup
