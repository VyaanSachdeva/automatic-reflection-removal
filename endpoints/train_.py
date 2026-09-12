"""Training API for UnReflectAnything."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

from os import PathLike


def train(
    config: Union[str, PathLike, Path] = "config/train.yaml",
    resume_run: Optional[str] = None,
    boot: bool = False,
    **overrides,
) -> None:
    """Run the training pipeline.

    This function trains the UnReflectAnything model using the specified
    configuration. It supports resuming from checkpoints and overriding
    config parameters via keyword arguments.

    Args:
        config: Path to the training configuration YAML file.
        resume_run: Run identifier to resume training from. If provided,
            training continues from the last checkpoint of the specified run.
        boot: If True, run in boot mode with minimal parameters for quick testing
            (batch_size=1, epochs=1, no_wandb=True).
        **overrides: Additional config overrides in the format PARAM=value.
            These override values from the config file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        RuntimeError: If training fails.
    """
    config_path = Path(config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    argv_backup = sys.argv
    new_argv = [sys.argv[0], "--config", str(config_path)]

    if resume_run:
        new_argv.extend(["--resume-run", resume_run])
    if boot:
        new_argv.append("--boot")

    for key, value in overrides.items():
        new_argv.append(f"--{key.upper()}={value}")

    try:
        sys.argv = new_argv
        import main

        main.run_pipeline(mode="train")
    finally:
        sys.argv = argv_backup
