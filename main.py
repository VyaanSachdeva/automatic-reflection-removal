#  MODULES AND DATASET LOADING
from dotenv import load_dotenv

load_dotenv()
import argparse
import os
import time
import torch
import torch.distributed as dist

import debugpy
from rich.traceback import install

from engine import Engine
from logger import get_logger
from utilities.system_ops import titlescreen
from typing import Dict, Any, Optional
from utilities.config import DISTRIBUTE_DDP, DISTRIBUTE_DP, DISTRIBUTE_SINGLEGPU

logger = get_logger(__name__).set_context("IMPORT")


def run_pipeline(mode: str = "train", config: Optional[Dict[str, Any]] = None) -> None:
    """
    Common pipeline for train and test modes.

    This is the main entry point that orchestrates the entire machine learning pipeline:
    - Environment setup and device detection
    - Configuration loading and processing
    - Model and dataset instantiation
    - Training or testing execution via Engine
    - Checkpoint management and evaluation

    The pipeline supports both training and testing modes with comprehensive
    logging, debugging capabilities, and optional experiment tracking.

    Args:
        mode (str): Operation mode, either 'train' or 'test'. Defaults to 'train'.
                   - 'train': Runs full training loop with validation and final testing
                   - 'test': Loads best checkpoint and runs evaluation only
        config (Optional[Dict[str, Any]]): Optional configuration dictionary to use
                                          instead of loading from file

    Returns:
        None

    Raises:
        ValueError: If mode is not 'train' or 'test'
        FileNotFoundError: If configuration file is not found
        RuntimeError: If model creation or dataset loading fails

    Note:
        This function handles:
        - CUDA device detection and setup
        - Multi-core CPU detection for data loading
        - Debug server setup if enabled
        - Experiment note management for reproducibility
        - VM auto-shutdown for cloud environments
    """
    # from utilities import *   # <--- this is not allowed within a function. Check if we need something from utilities
    import utilities.engine_initializers as initialize
    from utilities.config import (
        create_model_from_config,
        create_datasets_from_config,
        load_and_process_config,
        wrap_model_for_parallelization,
    )

    install(show_locals=False)

    # Argparse
    parser = argparse.ArgumentParser(description=f"{mode.capitalize()} the network")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--wait-debugger-attach",
        "-wd",
        action="store_true",
        help="Wait for debugger to attach",
    )
    parser.add_argument(
        "--record",
        "-r",
        action="store_true",
        help="Save the session for comparison",
    )
    parser.add_argument("--stop", "-s", action="store_true", help="Stop VM when done")
    parser.add_argument(
        "--boot",
        "-b",
        action="store_true",
        help="Run in boot mode with minimal parameters",
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        default=None,
        help="Path to the config file (optional positional)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=os.path.join("config", f"{mode}.yaml"),
        help="Path to the config file (overridden by positional if both given)",
    )
    parser.add_argument(
        "--resume-run",
        type=str,
        help="Resume training from an existing run. Provide the run name or run ID.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ddp",
        action="store_true",
        help="Force DistributedDataParallel (overrides DISTRIBUTE in config)",
    )
    group.add_argument(
        "--dp",
        action="store_true",
        help="Force DataParallel (overrides DISTRIBUTE in config)",
    )
    group.add_argument(
        "--single",
        action="store_true",
        help="Force single-GPU, non-distributed (overrides DISTRIBUTE). Short form.",
    )
    group.add_argument(
        "--singlegpu",
        action="store_true",
        help="Alias for --single (single-GPU, non-distributed).",
    )

    # Parse known and unknown arguments
    args, unknown = parser.parse_known_args()

    if args.debug:
        debug_port = int(os.getenv("DEBUGPY_PORT"))
        # Get machine hostname/IP for remote debugging
        import socket

        hostname = socket.gethostname()
        try:
            # Try to get the actual IP address
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "localhost"

        logger.info(f"Debug mode enabled on port {debug_port}")
        logger.info(
            f"Connect VSCode debugger to: {hostname} ({ip_address}:{debug_port})"
        )
        debugpy.listen(
            ("0.0.0.0", debug_port)
        )  # Listen on all interfaces for remote connections
        if args.wait_debugger_attach:
            logger.info("Waiting for debugger to attach...")
            debugpy.wait_for_client()

    # Load and process configuration (search cwd, then config/, then configs/)
    def _resolve_config_path(path: str) -> str:
        if os.path.isabs(path) and os.path.isfile(path):
            return path
        cwd = os.getcwd()
        basename = os.path.basename(path)
        for dir_name in ("", "config", "configs"):
            candidate = os.path.join(cwd, dir_name, basename) if dir_name else os.path.join(cwd, basename)
            if os.path.isfile(candidate):
                return candidate
        return path  # let load_and_process_config raise FileNotFoundError

    config_path_arg = args.config_file if args.config_file is not None else args.config
    CONFIG_PATH = _resolve_config_path(config_path_arg)
    config = load_and_process_config(
        config_path=CONFIG_PATH,
        config=config,
        unknown_args=unknown,
        boot_mode=args.boot,
    )
    # Optionally override DISTRIBUTE from CLI (--ddp, --dp, --single / --singlegpu)
    if args.ddp:
        config["DISTRIBUTE"] = DISTRIBUTE_DDP
    elif args.dp:
        config["DISTRIBUTE"] = DISTRIBUTE_DP
    elif args.single or args.singlegpu:
        config["DISTRIBUTE"] = DISTRIBUTE_SINGLEGPU

    # Resolve DISTRIBUTE and set device / DDP context
    distribute = config.get("DISTRIBUTE", DISTRIBUTE_SINGLEGPU)
    ddp_rank = None
    ddp_world_size = None

    if distribute == DISTRIBUTE_DDP:
        if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
            raise RuntimeError(
                "DISTRIBUTE is 'ddp' but not launched with torchrun. "
                "Run: torchrun --nproc_per_node=N train.py --config <config> ..."
            )
        dist.init_process_group(backend=config.get("DISTBACKEND", "gloo"))
        local_rank = int(os.environ["LOCAL_RANK"])
        ddp_rank = int(os.environ["RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        config["LOCAL_RANK"] = local_rank
        config["RANK"] = ddp_rank
        config["WORLD_SIZE"] = ddp_world_size
        DEVICE = torch.device(f"cuda:{local_rank}")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial logging: logger is DDP-aware and emits to console only on rank 0
    try:
        logger.info(titlescreen(), context="INFO")
    except Exception:
        logger.info("=" * 50, context="INFO")
        logger.info("UnReflectAnything - Reflection Removal Training", context="INFO")
        logger.info("=" * 50, context="INFO")

    logger.info(f"Torch Version: {torch.__version__}")
    logger.info(f"Python Version: {os.sys.version.split()[0]}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")

    try:
        cpu_affinity = os.sched_getaffinity(os.getpid())
        CPU_AFFINITY = len(list(cpu_affinity))
        logger.info(f"Cores available: {CPU_AFFINITY} {sorted(list(cpu_affinity))}")
    except Exception:
        logger.info("Couldn't get CPU affinity", context="INFO")

    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    # Run the appropriate function based on mode
    try:
        if mode == "train":
            # Check if we need to resume from an existing run
            if hasattr(args, "resume_run") and args.resume_run:
                logger.info(
                    f"Resuming training from run: {args.resume_run}", context="RESUME"
                )
                # Create datasets for training
                dataset = create_datasets_from_config(config)
                # Create model
                model = create_model_from_config(config, DEVICE)
                model = wrap_model_for_parallelization(model, config)

                # Discover existing run directories and latest checkpoint without
                # creating a new WandB run or new run directory.
                from utilities.run_resume import (
                    get_resume_info,
                    load_checkpoint_for_resume,
                )

                resume_info = get_resume_info(
                    args.resume_run, initialize.device_and_directories(config)["runs_dir"]
                )
                if resume_info is None:
                    logger.error(
                        f"No existing run matched resume identifier: {args.resume_run}"
                    )
                    return

                checkpoint_data = load_checkpoint_for_resume(
                    resume_info["latest_checkpoint"], DEVICE
                )
                if checkpoint_data is None:
                    logger.error(
                        f"Failed to load checkpoint for resume run: {args.resume_run}"
                    )
                    return

                resume_wandb_run_id = checkpoint_data.get("wandb_run_id", None)

                # Initialize engine in resume mode so that:
                # - Existing run directories are reused (no new RUN_DIR is created)
                # - WandB is resumed directly using the stored run ID (no new run)
                engine = Engine(
                    model=model,
                    dataset=dataset,
                    config=config,
                    no_wandb=config.get("NO_WANDB", False),
                    notes=config.get("NOTES", ""),
                    resume_run_id=resume_wandb_run_id,
                    will_resume=True,
                    resume_info=resume_info,
                    rank=ddp_rank,
                    world_size=ddp_world_size,
                )

                # Resume from the specified run (all ranks must succeed or all exit)
                resume_ok = engine.resume_from_run(args.resume_run)
                if distribute == DISTRIBUTE_DDP:
                    success_int = 1 if resume_ok else 0
                    sync_tensor = torch.tensor(
                        success_int, device=DEVICE, dtype=torch.int64
                    )
                    dist.all_reduce(sync_tensor, op=dist.ReduceOp.MIN)
                    resume_ok = sync_tensor.item() == 1
                if not resume_ok:
                    logger.info(
                        "Failed to resume training. Exiting.", context="RESUME"
                    )
                    return

                # Train the model (will start from the correct epoch)
                start_time = time.time()
                engine.trainloop()
                end_time = time.time()
                logger.info(f"Training time: {end_time - start_time:.2f} seconds")

                # Load best model and test
                engine.reinstantiate_model_from_checkpoint()
                engine.test()

            else:
                
                # Create datasets for training
                dataset = create_datasets_from_config(config)
                
                # Normal training (new run)
                # Create model
                model = create_model_from_config(config, DEVICE)
                model = wrap_model_for_parallelization(model, config)

                # Initialize engine
                engine = Engine(
                    model=model,
                    dataset=dataset,
                    config=config,
                    no_wandb=config.get("NO_WANDB", False),
                    resume_run_id=args.resume_run,
                    notes=config.get("NOTES", ""),
                    rank=ddp_rank,
                    world_size=ddp_world_size,
                )

                # Train the model
                engine.trainloop()

                # Load best model and test
                engine.reinstantiate_model_from_checkpoint()
                engine.test()

        elif mode == "test":
            # Resolve run identifier from config for test resume
            run_identifier = config.get("RUN", None)
            if run_identifier is None or (
                isinstance(run_identifier, str) and len(run_identifier.strip()) == 0
            ):
                logger.error(
                    "RUN name must be provided for test mode (via config or --RUN)"
                )
                return

            # Discover existing run directories without creating anything new
            from utilities.run_resume import get_resume_info

            resume_info = get_resume_info(
                run_identifier, initialize.device_and_directories(config)["runs_dir"]
            )
            if resume_info is None:
                logger.error(f"No existing run matched RUN name: {run_identifier}")
                return

            # Create model and datasets using the current config
            model = create_model_from_config(config, DEVICE)
            model = wrap_model_for_parallelization(model, config)
            dataset = create_datasets_from_config(config)

            # Initialize engine in resume mode, resuming the existing WandB run
            engine = Engine(
                model=model,
                dataset=dataset,
                config=config,
                notes=config.get("NOTES", ""),
                no_wandb=False,
                resume_run_id=None,  # let initializer resolve RUN by display name
                will_resume=True,
                resume_info=resume_info,
                rank=ddp_rank,
                world_size=ddp_world_size,
            )

            # Load best model from existing run and test
            engine.reinstantiate_model_from_checkpoint()
            engine.test()

        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        logger.error(f"Error in run_pipeline: {e}")
        raise e
