"""
Memory logging and cleanup helpers for the training Engine.
"""
from typing import Any, List, Optional

import torch

from utilities import system_ops


def log_memory_usage(
    logger: Any, context: str = "", memory_monitoring: bool = False
) -> None:
    """Log current GPU memory allocated/reserved if monitoring is enabled."""
    if memory_monitoring and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"GPU Memory - {context}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        )


def aggressive_memory_cleanup(
    logger: Any,
    aggressive_cleanup: bool,
    memory_monitoring: bool,
    exclude_vars: Optional[List[str]] = None,
) -> None:
    """Run gpuClean and optionally log freed memory."""
    if not aggressive_cleanup:
        return
    try:
        freed_count, memory_freed = system_ops.gpuClean(
            frame_up=1, exclude_vars=exclude_vars or [], verbose=False
        )
        if memory_monitoring and freed_count > 0:
            logger.info(
                f"Memory cleanup: Freed {freed_count} tensors, {memory_freed:.2f}MB"
            )
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")


def should_do_strategic_cleanup(
    phase: str, batch_idx: int, memory_cleanup_frequency: int
) -> bool:
    """Whether to run aggressive cleanup this batch."""
    return (
        batch_idx % memory_cleanup_frequency == 0
        or (
            phase == "Training"
            and batch_idx % (memory_cleanup_frequency * 2) == 0
        )
    )


def strategic_memory_cleanup(
    phase: str,
    batch_idx: int,
    memory_cleanup_frequency: int,
    logger: Any,
    aggressive_cleanup: bool,
    memory_monitoring: bool,
    exclude_vars: Optional[List[str]] = None,
) -> None:
    """Run aggressive cleanup when appropriate for this phase/batch."""
    if should_do_strategic_cleanup(phase, batch_idx, memory_cleanup_frequency):
        aggressive_memory_cleanup(
            logger, aggressive_cleanup, memory_monitoring, exclude_vars
        )
