import gc
import os
import shutil
import glob
from contextlib import contextmanager, nullcontext
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import math

import optimization
import utilities.engine_initializers as initialize
import wandb
from logger import get_logger
from losses import UnReflectLoss
from highlight_render import HighlightRender
from utilities.ablation import Ablation
from utilities.model import pixel_mask_to_patch_mask
from utilities import engine_helpers
from utilities import engine_memory
from utilities import engine_visualization as engine_viz
from utilities.system_ops import get_slurm_time_left_minutes

ablation = Ablation(False)


class Engine:
    def __init__(
        self,
        model: Union[nn.Module, str, None],
        dataset: dict,
        config: dict,
        notes: str = "",
        no_wandb: bool = False,
        resume_run_id: str = None,
        resume_info: Optional[dict] = None,
        will_resume: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Initializes the Engine object for polarization-based reflection removal training.

        Args:
            model (nn.Module): The model to be trained or model config.
            dataset (dict): Dictionary containing 'training', 'validation', and optionally 'test' datasets.
            config (dict): Dictionary containing config like BATCH_SIZE, LEARNING_RATE, etc.
            notes (str, optional): Additional notes for the training session. Defaults to "".
            no_wandb (bool): Whether to disable wandb logging.
            **kwargs: Additional keyword arguments.
        """
        # Set memory-efficient settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autograd.set_detect_anomaly(False)  # Disable for performance

        # Store configuration
        self.config = config
        self.config["NOTES"] = notes
        self.no_wandb = no_wandb
        # Bind ablation flag from config to both instance and module-level context
        try:
            enabled = bool(self.config.get("ABLATE", False))
        except Exception:
            enabled = False
        ablation.set(enabled)
        self.ablation = ablation
        # Mark if this Engine is expected to resume from an existing run
        # This must be set before any directory setup happens
        self._will_resume = bool(will_resume)
        # DDP: only set when config.DISTRIBUTE == "ddp"
        self._rank = rank
        self._world_size = world_size
        self._is_ddp = world_size is not None and world_size > 1
        self._distribute = config.get("DISTRIBUTE", "single")

        # Initialize device and directories
        device_dirs = initialize.device_and_directories(config)
        self.device = device_dirs["device"]
        self.RUNS_DIR = device_dirs["runs_dir"]

        # Function to set attributes from initialization functions
        def init(init_func, *args, **kwargs):
            result = init_func(*args, **kwargs)
            for key, value in result.items():
                setattr(self, key, value)
            return result

        # Initialize the models
        self.model = model
        # Initialize all components using engine_initializers
        init(initialize.dataloaders, dataset, config)
        # for i in range(len(dataset["Training"])):
        #     print(dataset["Training"][i]["raw"].shape)
        init(initialize.dimensions, self.training_dl, config)
        init(initialize.hyperparameters, config)
        init(initialize.optimizers, self.model, config)
        init(initialize.schedulers, self.optimizer, config, self.training_dl)
        init(initialize.transforms, self.height, self.width)
        # Wandb: initialized only on rank 0 when DDP (see utilities.engine_initializers.wandb)
        resume_wandb_run_id = resume_run_id
        init(initialize.wandb, config, self.model, notes, no_wandb, resume_wandb_run_id)
        init(initialize.tracking_metrics)

        # Skip directory setup during initialization if we're going to resume
        # The directories will be set up during resume_from_run()
        if not self._will_resume:
            if self._is_main_process() or not self._is_ddp:
                init(initialize.setup_run_directories, self.RUNS_DIR, self.wandb, False)
            if self._is_ddp:
                runname_list = [getattr(self, "runname", None)]
                dist.broadcast_object_list(runname_list, src=0)
                self.runname = runname_list[0]
                self.RUN_DIR = os.path.join(self.RUNS_DIR, self.runname)
                self.MODELS_DIR = os.path.join(self.RUN_DIR, "models")
                self.TEST_DIR = os.path.join(self.RUN_DIR, "tests")
                self.paths_file = os.path.join(self.RUN_DIR, "loadeddata.csv")
            init(
                initialize.earlystopping,
                self.earlystopping_patience,
                self.MODELS_DIR,
                self.runname,
            )
            self.config["name"] = self.runname
        else:
            # For resume mode, if resume_info is provided we bind the existing directories immediately
            if resume_info is not None:
                self.runname = os.path.basename(resume_info.get("run_dir"))
                self.RUN_DIR = resume_info.get("run_dir")
                self.MODELS_DIR = resume_info.get("models_dir")
                self.TEST_DIR = os.path.join(self.RUN_DIR, "tests")
                self.paths_file = os.path.join(self.RUN_DIR, "loadeddata.csv")
                self.config["name"] = self.runname
            else:
                # Temporary placeholder; will be updated upon resume
                self.config["name"] = "resuming"
        self.add_highlights = HighlightRender(height=self.height, width=self.width).to(
            self.device
        )

        # Save hyperparameters to json only when not resuming (rank 0 only when DDP)
        if not self._will_resume and self._is_main_process():
            initialize.save_hyperparameters_json(self.RUN_DIR, self.config)

        # Once the run name is set, we move all the log files to the run directory (rank 0 only when DDP)
        if not self._will_resume and self._is_main_process():
            TEMPORARY_LOG_DIR = os.path.join(self.RUNS_DIR, "temporary")
            if os.path.exists(TEMPORARY_LOG_DIR):
                for log_file in os.listdir(TEMPORARY_LOG_DIR):
                    if log_file.endswith(".log"):
                        shutil.move(
                            os.path.join(TEMPORARY_LOG_DIR, log_file),
                            os.path.join(self.RUN_DIR, log_file),
                        )

        # Initialize polarization-specific losses
        self.logger = get_logger(
            __name__,
            log_to_file=self._is_main_process(),
            relative_log_dir=self.RUN_DIR,
        )
        self.loss = UnReflectLoss(
            weight_specular_loss=self.config.SPECULAR_LOSS_WEIGHT,
            weight_diffuse_loss=self.config.DIFFUSE_LOSS_WEIGHT,
            weight_highlight_loss=self.config.HIGHLIGHT_LOSS_WEIGHT,
            weight_image_reconstruction=self.config.IMAGE_RECONSTRUCTION_LOSS_WEIGHT,
            # Highlight regression loss parameters
            hlreg_w_l1=float(self.config.get("HLREG_W_L1", 1.0)),
            hlreg_use_charb=bool(self.config.get("HLREG_USE_CHARB", True)),
            hlreg_w_dice=float(self.config.get("HLREG_W_DICE", 0.2)),
            hlreg_w_ssim=float(self.config.get("HLREG_W_SSIM", 0.0)),
            hlreg_w_grad=float(self.config.get("HLREG_W_GRAD", 0.0)),
            hlreg_w_tv=float(self.config.get("HLREG_W_TV", 0.0)),
            hlreg_balance_mode=self.config.get("HLREG_BALANCE_MODE", "none"),
            hlreg_pos_weight=float(self.config.get("HLREG_POS_WEIGHT", 1.0)),
            hlreg_focal_gamma=float(self.config.get("HLREG_FOCAL_GAMMA", 0.0)),
            # Highlight rendering parameters
            highlight_color=tuple(self.config.get("HIGHLIGHT_COLOR", [1.0, 1.0, 1.0])),
            clamp_reconstruction=bool(self.config.get("CLAMP_RECONSTRUCTION", True)),
            # Context and regularization weights
            weight_seam=float(self.config.get("WEIGHT_SEAM", 0.5)),
            ring_dilate_kernel=int(self.config.get("RING_DILATE_KERNEL", 7)),
            seam_use_charb=bool(self.config.get("SEAM_USE_CHARB", True)),
            seam_weight_grad=float(self.config.get("SEAM_WEIGHT_GRAD", 0.2)),
            # Token-space loss parameters
            weight_token_inpaint=float(
                self.config.get("TOKEN_INPAINT_LOSS_WEIGHT", 1.0)
            ),
            token_feat_alpha=float(self.config.get("TOKEN_FEAT_ALPHA", 0.5)),
            # Diffuse highlight penalty parameters
            weight_diffuse_highlight_penalty=float(
                self.config.get("WEIGHT_DIFFUSE_HIGHLIGHT_PENALTY", 0.0)
            ),
            diffuse_hl_threshold=float(self.config.get("DIFFUSE_HL_THRESHOLD", 0.7)),
            diffuse_hl_use_charb=bool(self.config.get("DIFFUSE_HL_USE_CHARB", True)),
            diffuse_hl_penalty_mode=self.config.get(
                "DIFFUSE_HL_PENALTY_MODE", "brightness"
            ),
            diffuse_hl_target_brightness=self.config.get(
                "DIFFUSE_HL_TARGET_BRIGHTNESS", None
            ),
            diffuse_hl_use_luminance=bool(
                self.config.get("DIFFUSE_HL_USE_LUMINANCE", False)
            ),
        ).to(self.device)

        # Memory management settings for optimal GPU memory usage
        self.memory_cleanup_frequency = config.get(
            "MEMORY_CLEANUP_FREQUENCY", 5
        )  # Clean every N batches
        self.aggressive_cleanup = config.get(
            "AGGRESSIVE_MEMORY_CLEANUP", True
        )  # Use gpuClean utility
        self.memory_monitoring = config.get(
            "MEMORY_MONITORING", False
        )  # Log memory usage

        # training_sampler is set by init(dataloaders) when DDP; otherwise None
        self.training_sampler = getattr(self, "training_sampler", None)

    def _is_main_process(self) -> bool:
        """True if this process should log, save checkpoints, and run test."""
        return self._rank is None or self._rank == 0

    def _unwrap_model(self) -> nn.Module:
        """Return the underlying model (strip DDP or DataParallel wrapper)."""
        if self._distribute in ("dp", "ddp"):
            return self.model.module
        return self.model

    def composite_specular_diffuse(
        self, specular: torch.Tensor, diffuse: torch.Tensor
    ) -> torch.Tensor:
        """
        Composite specular and diffuse components into a reconstructed image.

        Args:
            specular (torch.Tensor): Specular component [B, C, H, W]
            diffuse (torch.Tensor): Diffuse component [B, C, H, W]

        Returns:
            torch.Tensor: Reconstructed image [B, 3, H, W]
        """
        if self.memory_monitoring:
            engine_memory.log_memory_usage(
                self.logger, "Before compositing", self.memory_monitoring
            )
        return engine_helpers.composite_specular_diffuse(specular, diffuse)

    def trainloop(self):
        """
        The main training loop that runs through all epochs, trains the model,
        validates it, and handles early stopping and saving of the model.
        """
        # Determine starting epoch (for resume functionality)
        start_epoch = getattr(self, "start_epoch", 0)

        if start_epoch > 0 and self._is_main_process():
            self.logger.info(
                f"Starting training loop from epoch index {start_epoch} (display: epoch {start_epoch + 1}/{self.epochs})",
                context="TRAINING",
            )

        for e in range(start_epoch, self.epochs):
            ### TRAINING + VALIDATION FOR EACH EPOCH
            self.train()  # Train the model for one epoch
            is_overfitting = self.validate()  # Train the model for one epoch

            if self._is_main_process():
                self.csv_log_metrics()  # Log the metrics to csv

            # Save checkpoint every few epochs (rank 0 only when DDP)
            if (e + 1) % self.config.get("SAVE_INTERVAL", 10) == 0:
                self.save_checkpoint(e)

            ### BREAK IF EARLYSTOP
            if is_overfitting == "EARLYSTOP":
                break  # Exit the training loop if early stopping condition is met

        # Log locations of important data at the end of training (rank 0 only when DDP)
        if self._is_main_process():
            self.logger.info("TRAINING COMPLETE", context="SAVE")
            self.logger.info(
                f"Run directory: {os.path.abspath(self.RUN_DIR)}", context="SAVE"
            )
            self.logger.info(
                f"Checkpoints  : {os.path.abspath(self.MODELS_DIR)}", context="SAVE"
            )
            self.logger.info("Metrics      :", context="SAVE")
            self.logger.info(
                f"Training     : {os.path.abspath(os.path.join(self.RUN_DIR, 'training_metrics.csv'))}",
                context="SAVE",
            )
            self.logger.info(
                f"Validation   : {os.path.abspath(os.path.join(self.RUN_DIR, 'validation_metrics.csv'))}",
                context="SAVE",
            )

            # Remove unused IMAGES_DIR if it exists and is empty
            images_dir = os.path.join(self.RUN_DIR, "images")
            if os.path.exists(images_dir) and not os.listdir(images_dir):
                try:
                    os.rmdir(images_dir)
                    self.logger.info(
                        f"Removed unused directory: {images_dir}", context="SAVE"
                    )
                except OSError:
                    pass

    def train(self):
        """Training phase for one epoch"""
        return self.run_epoch(phase="Training")

    def validate(self):
        """Validation phase for one epoch"""
        result = self.run_epoch(phase="Validation")

        # Early stopping logic
        if result is not None:
            self.step["epoch"] += 1  # Increasing epoch counter
            self.LRschedulerPlateau.step(float(result))
            if self._is_main_process():
                self.earlystopping(
                    float(result),
                    self.model,
                    self.step["epoch"] - 1,
                    self.optimizer,
                    self.config,
                    self.wandb,
                )
            if self._is_ddp:
                # Broadcast early_stop from rank 0 to all ranks
                early_stop_tensor = torch.tensor(
                    1
                    if (self._is_main_process() and self.earlystopping.early_stop)
                    else 0,
                    device=self.device,
                    dtype=torch.int64,
                )
                dist.broadcast(early_stop_tensor, src=0)
                if early_stop_tensor.item() != 0:
                    self.earlystopping.early_stop = True
            if self.earlystopping.early_stop:
                if self._is_main_process():
                    self.logger.info(
                        ">> [EARLYSTOPPING]: Patience Reached, Stopping Training",
                        context="TRAINING",
                    )
                return "EARLYSTOP"
            return "IMPROVED"
        return "CONTINUE"

    def test(self):
        """Test phase"""
        # Persistent test index: auto test after training is idx 0
        test_idx = self._load_and_increment_test_index()
        result = self.run_epoch(phase="Test", test_idx=test_idx)
        if self.wandb is not None:
            self.log_tests()
        return result

    def log_tests(self):
        """
        Logs the test metrics to Weights and Biases.
        """
        self.logger.info(">> TEST REPORT", context="TEST")
        self.logger.info(self.metrics["Test"].describe(), context="TEST")
        self.metrics["Test"].to_csv(os.path.join(self.RUN_DIR, "test_metrics.csv"))
        if self.wandb:
            self.wandb.log(
                {"Test/Summary": wandb.Table(dataframe=self.metrics["Test"])}
            )

        # Log locations of important data
        self.logger.info(">> RUN DATA LOCATIONS", context="SAVE")
        self.logger.info(
            f"Run data directory: {os.path.abspath(self.RUN_DIR)}", context="SAVE"
        )
        self.logger.info(
            f"Models saved at: {os.path.abspath(self.MODELS_DIR)}", context="SAVE"
        )
        self.logger.info("Metrics CSV files:", context="SAVE")
        self.logger.info(
            f"  - Training: {os.path.abspath(os.path.join(self.RUN_DIR, 'training_metrics.csv'))}",
            context="SAVE",
        )
        self.logger.info(
            f"  - Validation: {os.path.abspath(os.path.join(self.RUN_DIR, 'validation_metrics.csv'))}",
            context="SAVE",
        )
        self.logger.info(
            f"  - Test: {os.path.abspath(os.path.join(self.RUN_DIR, 'test_metrics.csv'))}",
            context="SAVE",
        )

        # Log WandB URLs again for convenience
        if hasattr(self.wandb, "url") and self.wandb.url:
            self.logger.info(f"WandB run URL: {self.wandb.url}", context="WANDB")
            project_url = self.wandb.url.rsplit("/", 1)[0]
            self.logger.info(f"WandB project URL: {project_url}", context="WANDB")

    def csv_log_metrics(self):
        """Save metrics to CSV files"""
        if not self.metrics["Training"].empty:
            self.metrics["Training"].to_csv(
                os.path.join(self.RUN_DIR, "training_metrics.csv")
            )
        if not self.metrics["Validation"].empty:
            self.metrics["Validation"].to_csv(
                os.path.join(self.RUN_DIR, "validation_metrics.csv")
            )

    def console_log_metrics(
        self,
        stage,
        epoch=None,
        batch_idx=None,
        dataloader_len=None,
        extra_info=None,
    ):
        """
        Print metrics and status information for training, validation, or test.

        Parameters:
        -----------
        stage : str
            The current stage ('Training', 'Validation', or 'Test').
        epoch : int, optional
            Current epoch number.
        batch_idx : int, optional
            Current batch index.
        dataloader_len : int, optional
            Length of the dataloader being used.
        extra_info : str, optional
            Additional information to display in the phase indicator.
        """

        # Simple alignment function (since we don't have the original utilities)
        def align(text, width, direction="left"):
            if direction == "left":
                return f"{text:<{width}}"
            elif direction == "right":
                return f"{text:>{width}}"
            else:  # center
                return f"{text:^{width}}"

        epoch_batch_info = align(
            f"E {str(epoch + 1)}/{self.epochs} ", 10, "right"
        ) + align(f"B {str(batch_idx + 1)}/{dataloader_len} ", 10, "left")

        # Note: extra_info is used in console logging but phase_indicator was unused

        # Print header with run name and status information
        if "offline" in self.runname:
            printedrunname = "run"
        else:
            printedrunname = f"{self.runname.split('-')[0][0]}{self.runname.split('-')[1][0]}{self.runname.split('-')[2]}"

        metricstring = align(f"{printedrunname}:", 6, "right") + epoch_batch_info

        # Generate metrics string
        metrs = ""
        # Print metrics from the appropriate metrics dictionary
        if stage in self.metrics.keys():
            # First print the Loss column if it exists
            if (
                "Loss" in self.metrics[stage].columns
                and self.metrics[stage]["Loss"].iloc[-1] is not None
            ):
                metrs += (
                    "[yellow]Loss[/yellow]"
                    + "="
                    + align(
                        f"{self.metrics[stage]['Loss'].iloc[-1]:.4f}",
                        6,
                        "left",
                    )
                    + " "
                )

            # Then print other columns that don't have a "/" in their name
            for m in self.metrics[stage].columns:
                if (
                    m != "Loss"
                    and "/" not in m
                    and self.metrics[stage][m].iloc[-1] is not None
                    and not m.startswith("Step/")  # Skip step metrics
                    and not m.startswith(
                        "HyperParameters/"
                    )  # Skip hyperparameter metrics
                ):
                    # Use full metric name for better readability
                    display_name = m if len(m) <= 6 else m[:6]
                    metrs += (
                        f"[yellow]{display_name}[/yellow]"
                        + "="
                        + align(
                            f"{self.metrics[stage][m].iloc[-1]:.4f}",
                            6,
                            "left",
                        )
                        + " "
                    )
        self.logger.info(metricstring + metrs, context=stage.upper())

    def log_loaded_paths(self, paths, phase):
        """Log loaded file paths for debugging"""
        if hasattr(self, "paths_file"):
            with open(self.paths_file, mode="a") as file:
                file.write(f"{self.step[f'{phase}_batch']},{paths}\n")

    def backward_pass(
        self,
        loss_tensor,
        accumulate_gradients=False,
        phase="Training",
        submodules_to_monitor=None,
    ):
        """
        Performs the backward pass, including gradient calculation, clipping, and optimization steps.
        Optimized for memory efficiency with strategic cleanup.

        Args:
            loss_tensor (torch.Tensor): The loss tensor to backpropagate
            accumulate_gradients (bool): If True, will update weights after backpropagation
                                        assuming gradient accumulation is complete
            phase (str): Current phase ("Training", "Validation", "Test")
            submodules_to_monitor (dict, optional): Dictionary mapping submodule names to submodule objects
                                                   for separate gradient/weight norm monitoring

        Returns:
            dict: A dictionary containing gradient norms, weight norms, and error status
        """
        # Initialize return values
        grad_norm = np.nan
        weight_norm = np.nan
        submodule_norms = {}

        if phase != "Training":
            loss_tensor.detach()
            torch.cuda.empty_cache()
            return {
                "ERROR_IN_BACKWARD_PASS": False,
                "grad_norm": grad_norm,
                "weight_norm": weight_norm,
                "submodule_norms": submodule_norms,
            }

        # Log memory usage before backward pass if monitoring is enabled
        if self.memory_monitoring:
            self._log_memory_usage("Before backward pass")

        ERROR_IN_BACKWARD_PASS = False
        NON_FINITE_GRAD = False
        try:
            loss_tensor.backward()
        except RuntimeError as e:
            self.logger.error(
                f">> [ERROR]: {e} - Skipping batch {self.step['Training_batch']} in epoch {self.step['epoch']}"
            )
            ERROR_IN_BACKWARD_PASS = True

        # Gradient clipping with memory optimization
        if (
            self.config.GRADIENT_CLIPPING_MAX_NORM is not None
            and not ERROR_IN_BACKWARD_PASS
        ):
            try:
                # Calculate gradient and weight norms for the whole model
                grad_norm, weight_norm = optimization.get_norms(self.model.parameters())
                # Guard against NaN/Inf gradients: a single optimizer.step() with
                # non-finite grads silently poisons every weight (observed as
                # MODEL_GradNorm=NaN in unique-sponge-1115). clip_grad_norm_ does NOT
                # sanitize NaN, and the step below runs unconditionally otherwise.
                NON_FINITE_GRAD = not np.isfinite(grad_norm)

                # Calculate norms for specific submodules if requested
                if submodules_to_monitor is not None:
                    for submodule_name, submodule in submodules_to_monitor.items():
                        if submodule is None:
                            continue
                        try:
                            sub_grad_norm, sub_weight_norm = optimization.get_norms(
                                submodule.parameters()
                            )
                            submodule_norms[submodule_name] = {
                                "grad_norm": sub_grad_norm,
                                "weight_norm": sub_weight_norm,
                            }
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to compute norms for submodule {submodule_name}: {e}"
                            )
                            submodule_norms[submodule_name] = {
                                "grad_norm": np.nan,
                                "weight_norm": np.nan,
                            }

                if self.config.GRADIENT_CLIPPING_MAX_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.GRADIENT_CLIPPING_MAX_NORM,
                    )
            except Exception as e:
                self.logger.warning(f"Gradient clipping failed: {e}")

        # Step only if warmup phase is finished and we are backpropagating the accumulated gradients
        if accumulate_gradients:
            if not ERROR_IN_BACKWARD_PASS and not NON_FINITE_GRAD:
                self.optimizer.step()
                self.optimizer.zero_grad()
            elif NON_FINITE_GRAD:
                # Discard this update and clear the poisoned grads so they don't
                # accumulate into the next step.
                self.logger.warning(
                    f">> [WARN]: Non-finite gradient norm ({grad_norm}) at batch "
                    f"{self.step['Training_batch']} in epoch {self.step['epoch']} - "
                    f"skipping optimizer step and clearing gradients."
                )
                self.optimizer.zero_grad()
            self.LRscheduler.step()

        # Aggressive memory cleanup after backward pass
        loss_tensor.detach()
        del loss_tensor

        # Clean up gradients and intermediate computations
        if self.aggressive_cleanup:
            # Only clear gradients if we actually performed an optimizer step; otherwise
            # leave them to accumulate for gradient accumulation.
            if accumulate_gradients:
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad = None

            # Force garbage collection and cache clearing
            torch.cuda.empty_cache()
            gc.collect()

        return {
            "ERROR_IN_BACKWARD_PASS": ERROR_IN_BACKWARD_PASS,
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
            "submodule_norms": submodule_norms,
        }

    def switch_optimizer(self, current_epoch):
        """
        Switches the optimizer if the current epoch matches the switch epoch and
        the bootstrap and refining optimizers are different.

        Args:
            current_epoch (int): The current epoch number

        Returns:
            bool: True if the optimizer was switched, False otherwise
        """
        if (
            current_epoch == self.switch_optimizer_epoch
            and self.optimizer_bootstrap_name != self.optimizer_refining_name
        ):
            self.logger.info(
                f">> [OPTIMIZER]: "
                f"Switching from [{self.optimizer_bootstrap_name}] to [{self.optimizer_refining_name}]"
            )
            self.in_optswitch_phase = True
            self.optimizer = getattr(optimization, self.optimizer_refining_name)(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            return True
        return False

    def run_epoch(self, phase: str, test_idx: Optional[int] = None) -> Optional[float]:
        """
        Run one epoch of training, validation, or test.
        Adapted for polarization-based reflection removal with memory optimizations.

        Args:
            phase: "Training", "Validation", or "Test"

        Returns:
            Average loss for the epoch (if applicable)
        """
        # Phase setup
        is_training = phase == "Training"

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        # Get dataset from the initialized dataset structure
        dataset = self.dataset[phase]

        if dataset is None:
            self.logger.warning(
                f"No dataset available for {phase}", context=phase.upper()
            )
            return None

        # DDP: run Test phase only on rank 0
        if phase == "Test" and self._is_ddp and not self._is_main_process():
            return None

        # When DDP, use stored training_dl/validation_dl and set_epoch for training
        if self._is_ddp and phase in ("Training", "Validation"):
            if phase == "Training" and self.training_sampler is not None:
                self.training_sampler.set_epoch(self.step["epoch"])
            dataloader = self.training_dl if phase == "Training" else self.validation_dl
        else:
            cpu_affinity = os.sched_getaffinity(os.getpid())
            if self.config.get("NUM_WORKERS", "auto") == "auto":
                NUM_WORKERS = int(math.floor(0.9 * len(list(cpu_affinity))))
            else:
                NUM_WORKERS = self.config.NUM_WORKERS
            NUM_WORKERS = int(math.floor(0.9 * len(list(cpu_affinity))))
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=NUM_WORKERS
                if self.config.NUM_WORKERS == "auto"
                else self.config.NUM_WORKERS,
                drop_last=True,
                pin_memory=self.config.PIN_MEMORY,
                prefetch_factor=self.config.PREFETCH_FACTOR,
                shuffle=self.config.SHUFFLE,
            )

        if len(dataloader) == 0:
            self.logger.warning(
                "Empty dataloader, skipping epoch.", context=phase.upper()
            )
            return None

        epoch_losses = []
        images_logged = False

        # Switch optimizer if needed (for training)
        if is_training:
            self.switch_optimizer(self.step["epoch"])

        base_lr = self.optimizer.param_groups[0]["lr"]

        # Get image logging frequency from config
        image_log_interval = self.config.get("IMAGE_LOG_INTERVAL", 20)

        with self.choose_if_grad(phase):
            for batch_idx, sample in enumerate(dataloader):
                # Strategic memory cleanup at batch start
                if batch_idx == 0 or batch_idx % self.memory_cleanup_frequency == 0:
                    torch.cuda.empty_cache()
                    if self.aggressive_cleanup:
                        gc.collect()

                # Calculate step for warmup logic
                step = self.step["epoch"] * len(dataloader) + batch_idx

                # Determine if we should log images on this batch
                log_images_this_batch = (
                    batch_idx > 0
                    and batch_idx % image_log_interval == 0
                    and image_log_interval > 1
                ) or (batch_idx == len(dataloader) - 1 and not images_logged)

                # Warmup logic
                if is_training and step < self.warmup_steps:
                    warmup_factor = step / self.warmup_steps
                    current_lr = base_lr * warmup_factor
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = current_lr

                ### Add synthetic highlights (Blinn-Phong on MoGe normals).
                # During Validation/Test we FREEZE the highlight distribution so the
                # eval metric is comparable across runs and epochs. The swept training
                # params (HIGHLIGHT_FALLOFF / HIGHLIGHT_DENSITY, and any range-valued
                # SURFACE_ROUGHNESS / INTENSITY) must NOT leak into eval-time difficulty
                # -- otherwise a run trained on harder highlights is graded on a harder
                # eval set and looks worse for the wrong reason. We pin the eval params
                # to fixed values and seed the RNG per batch so identical highlights are
                # drawn every epoch and in every sweep run, then restore RNG afterwards.
                freeze_val_hl = (not is_training) and bool(
                    self.config.get("FREEZE_VAL_HIGHLIGHTS", False)
                )
                if freeze_val_hl:
                    hl_falloff = self.config.get("VAL_HIGHLIGHT_FALLOFF", 0.0)
                    hl_density = self.config.get("VAL_HIGHLIGHT_DENSITY", 0.0)
                    hl_roughness = self.config.get(
                        "VAL_SURFACE_ROUGHNESS", self.config.SURFACE_ROUGHNESS
                    )
                    hl_intensity = self.config.get("VAL_INTENSITY", self.config.INTENSITY)
                    _val_seed = int(self.config.get("VAL_HIGHLIGHT_SEED", 1234)) + batch_idx
                    _cpu_rng_state = torch.get_rng_state()
                    _cuda_rng_state = (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    )
                    torch.manual_seed(_val_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(_val_seed)
                else:
                    hl_falloff = self.config.get("HIGHLIGHT_FALLOFF", 0.0)
                    hl_density = self.config.get("HIGHLIGHT_DENSITY", 1.0)
                    hl_roughness = self.config.SURFACE_ROUGHNESS
                    hl_intensity = self.config.INTENSITY

                # Small, sharp, high-gradient specular blobs (normal perturbation via
                # add_geometric_roughness_torch). This synthesizes the "concentrated
                # pure-white blob with no smooth falloff" morphology that smooth
                # scene-wide Blinn-Phong lobes never produce. Disabled by default
                # (N_BLOBS=0 -> function not called, legacy behavior). Applied ONLY to
                # training: the blob RNG is not torch-seedable, so injecting it into a
                # frozen/seeded eval would break cross-run comparability.
                _geo = self.config.get("GEOMETRIC_HIGHLIGHTS", {}) or {}
                if freeze_val_hl or int(_geo.get("N_BLOBS", 0)) <= 0:
                    geo_kwargs = {}
                else:
                    geo_kwargs = dict(
                        n_blobs=int(_geo.get("N_BLOBS", 0)),
                        avg_blob_size=float(_geo.get("AVG_BLOB_SIZE", 0.05)),
                        size_unit=str(_geo.get("SIZE_UNIT", "fraction")),
                        size_spread=float(_geo.get("SIZE_SPREAD", 0.8)),
                        elongation_bias=float(_geo.get("ELONGATION_BIAS", 0.6)),
                        roughness_strength=float(_geo.get("ROUGHNESS_STRENGTH", 1.5)),
                    )

                # Large contiguous saturated highlights (make_large_highlight_mask):
                # composite a big irregular plateau into H so the token inpainter is
                # trained on LARGE holes whose interior has no local context (the
                # gray-center failure case). Unlike GEOMETRIC_HIGHLIGHTS these draw
                # from the global torch RNG, so they ARE reproducible under frozen
                # eval — VAL_LARGE_HIGHLIGHTS lets a frozen sweep grade large-hole
                # removal instead of being blind to it. PROB=0 (default) disables.
                if freeze_val_hl:
                    _lhl = self.config.get("VAL_LARGE_HIGHLIGHTS", {}) or {}
                else:
                    _lhl = self.config.get("LARGE_HIGHLIGHTS", {}) or {}
                if float(_lhl.get("PROB", 0.0)) > 0.0:
                    lhl_kwargs = dict(
                        large_highlight_prob=float(_lhl.get("PROB", 0.0)),
                        large_highlight_count=int(_lhl.get("COUNT", 1)),
                        large_highlight_size=_lhl.get("SIZE", [0.12, 0.38]),
                        large_highlight_elongation=float(_lhl.get("ELONGATION", 0.5)),
                        large_highlight_core_frac=float(_lhl.get("CORE_FRAC", 0.22)),
                        large_highlight_peak=_lhl.get("PEAK", [0.9, 1.5]),
                        large_highlight_edge_wobble=float(_lhl.get("EDGE_WOBBLE", 0.25)),
                    )
                else:
                    lhl_kwargs = {}

                random_light_pos = self.add_highlights.sample_light_source(
                    dist_to_camera=self.config.LIGHT_DISTANCE_RANGE,
                    left_right_angle=self.config.LIGHT_LEFT_RIGHT_ANGLE,
                    above_below_angle=self.config.LIGHT_ABOVE_BELOW_ANGLE,
                    batch_size=self.batch_size,
                    device=self.device,
                )
                highlight_result = self.add_highlights(
                    rgb=sample["diffuse"].to(self.device, non_blocking=True),
                    light_pos=random_light_pos,
                    surface_roughness=hl_roughness,
                    intensity=hl_intensity,
                    highlight_falloff=hl_falloff,
                    highlight_density=hl_density,
                    **geo_kwargs,
                    **lhl_kwargs,
                    return_dataset_highlights=True,
                    dataset_highlight_dilation=self.config.DATASET_HIGHLIGHT_DILATION,
                    dataset_highlight_threshold=self.config.DATASET_HIGHLIGHT_THRESHOLD,
                    dataset_highlight_use_luminance=bool(
                        self.config.get("DATASET_HIGHLIGHT_USE_LUMINANCE", True)
                    ),
                )
                # Restore RNG so frozen-eval seeding never perturbs the training stream.
                if freeze_val_hl:
                    torch.set_rng_state(_cpu_rng_state)
                    if _cuda_rng_state is not None:
                        torch.cuda.set_rng_state_all(_cuda_rng_state)

                dataset_highlights_soft_mask = highlight_result[
                    "dataset_highlights_soft_mask"
                ]
                synthetic_highlights_soft_mask = highlight_result["highlight"]

                ### SUPERVISION MASKS: Control which pixels and patches provide supervision
                # 1 if pixel provides supervision, 0 if not
                pixel_supervision_mask = highlight_result["pixel_supervision_mask"]

                # 1 if patch provides supervision and is to be included in the loss computation, 0 if to be excluded
                patch_supervision_mask = pixel_mask_to_patch_mask(
                    dataset_highlights_soft_mask,
                    patch_size=16,
                    threshold=self.config.DATASET_HIGHLIGHT_SUPERVISION_THRESHOLD,
                    invert=True,
                )

                ### HOLE MASKS: Control which pixels and patches are to be inpainted
                pixel_inpaint_soft_mask = torch.clamp(
                    synthetic_highlights_soft_mask + dataset_highlights_soft_mask,
                    0,
                    1,
                )
                # 1 if the pixel needs to be inpainted, 0 if not
                pixel_inpaint_mask = (
                    pixel_inpaint_soft_mask
                    > self.config.DATASET_HIGHLIGHT_SUPERVISION_THRESHOLD
                ).bool()
                # 1 if the patch needs to be inpainted, 0 if not
                patch_inpaint_mask = pixel_mask_to_patch_mask(
                    pixel_inpaint_mask,
                    patch_size=16,
                    threshold=self.config.DATASET_HIGHLIGHT_SUPERVISION_THRESHOLD,
                    invert=False,
                )

                # Token inpainter ground truth: run with unwrapped model under no_grad so that
                # only one forward per step goes through DDP and participates in loss. Otherwise
                # DDP sees a forward that does not use all parameters (just_extract_tokens=True
                # uses only the encoder) and raises "Expected to have finished reduction...".
                diffuse_img = sample["diffuse"].to(self.device, non_blocking=True)
                with torch.no_grad():
                    raw_model = self._unwrap_model()
                    diffuse_teacher_tokens = raw_model(
                        diffuse_img, just_extract_tokens=True
                    )

                ### Constructing ground truth dict
                rgb_highlighted = highlight_result["rgb_highlighted"]

                if "specular" in sample:
                    specular = sample["specular"].to(self.device, non_blocking=True)
                else:
                    specular = None

                # Ground truth diffuse: original RGB (contains real highlights)
                diffuse = sample["diffuse"].to(self.device, non_blocking=True)

                gt_decomposition = {
                    "diffuse": diffuse,  # Contains real highlights, but masked during loss
                    "rgb_highlighted": rgb_highlighted,
                    "specular": specular,
                    "highlight": pixel_inpaint_soft_mask,
                    "tokens_teacher": diffuse_teacher_tokens,
                }
                del sample  # Clean up. All necessary data is already on GPU.

                # Log memory usage before forward pass if monitoring
                if self.memory_monitoring and batch_idx % 10 == 0:
                    self._log_memory_usage(f"Before forward pass - batch {batch_idx}")

                if self.ablation:
                    print("This is an ablation block")

                ### Forward pass
                model_input = {
                    "rgb": gt_decomposition["rgb_highlighted"],
                    # "inpaint_mask_override": pixel_inpaint_mask,
                    "inpaint_mask_threshold": self.config.INPAINT_MASK_THRESHOLD,
                    "inpaint_mask_dilation": self.config.INPAINT_MASK_DILATION,
                    "inpaint_feed_raw_mask": self.config.get(
                        "INPAINT_FEED_RAW_MASK", False
                    ),
                }
                # if self._distribute == "dp":
                #     pred_decomposition = self.model(
                #         model_input["rgb"],
                #         model_input["inpaint_mask_override"],
                #         model_input["inpaint_mask_threshold"],
                #         model_input["inpaint_mask_dilation"],
                #     )
                # else:
                pred_decomposition = self.model(model_input)

                ### COMPUTE LOSS FUNCTION
                losses = self.loss(
                    prediction=pred_decomposition,
                    ground_truth=gt_decomposition,
                    pixel_supervision_mask=pixel_supervision_mask,
                    pixel_inpaint_mask=pixel_inpaint_mask,
                    patch_supervision_mask=patch_supervision_mask,
                    patch_inpaint_mask=patch_inpaint_mask,
                )
                loss_value = losses["total"]

                # Compositing the reconstructed image - for visualization purposes
                if "diffuse" in pred_decomposition:
                    pred_decomposition["rgb_highlighted"] = self.loss.reconstruct_image(
                        pred_decomposition
                    )
                # Adding the loss mask to the gt_decomposition to log it on wandb
                if phase == "Training":
                    gt_decomposition["supervision_mask"] = pixel_supervision_mask
                    gt_decomposition["masked_diffuse"] = (
                        diffuse * pixel_supervision_mask
                    )
                ### Backward pass
                backward_output = None
                if is_training:
                    try:
                        # Check if we should accumulate gradients
                        accumulate_gradients = (
                            batch_idx + 1
                        ) % self.gradient_accumulation_steps == 0

                        # Use the backward_pass method
                        backward_output = self.backward_pass(
                            loss_value,
                            accumulate_gradients=accumulate_gradients,
                            phase=phase,
                            submodules_to_monitor={
                                "highlight_decoder": getattr(
                                    self._unwrap_model().decoders,
                                    "highlight",
                                    None,
                                ),
                                "diffuse_decoder": getattr(
                                    self._unwrap_model().decoders,
                                    "diffuse",
                                    None,
                                ),
                                "specular_decoder": getattr(
                                    self._unwrap_model().decoders,
                                    "specular",
                                    None,
                                ),
                                "dinov3": self._unwrap_model().dinov3,
                                "token_inpaint": getattr(
                                    self._unwrap_model(),
                                    "token_inpaint",
                                    None,
                                ),
                            },
                        )

                    except Exception as e:
                        self.logger.error(
                            f"Error in backward pass: {e}", context=phase.upper()
                        )
                        continue

                # Track metrics
                epoch_losses.append(loss_value.item())
                self.step[f"{phase}_batch"] += 1

                # Update metrics dataframe
                metrics = {
                    "Loss": loss_value.item(),
                    "HyperParameters/LR": self.optimizer.param_groups[0]["lr"],
                    f"Step/{'val' if phase == 'Validation' else ''}batch": self.step[
                        f"{phase}_batch"
                    ],
                    f"Step/{'idx' if phase == 'Test' else 'epoch'}": self.step["epoch"],
                }
                if phase == "Test" and test_idx is not None:
                    metrics["Step/test_idx"] = test_idx
                    metrics["Index"] = float(test_idx)

                # Add individual loss components if available
                if isinstance(losses, dict):
                    for loss_name, loss_val in losses.items():
                        if isinstance(loss_val, torch.Tensor) and loss_name != "total":
                            # Use the loss name directly (without "Loss_" prefix) for better display
                            metrics[loss_name] = loss_val.item()

                # Add gradient information if available
                if (
                    backward_output is not None
                    and backward_output.get("grad_norm") is not None
                ):
                    metrics["Gradients/MODEL_GradNorm"] = backward_output["grad_norm"]
                    metrics["Gradients/MODEL_WeightNorm"] = backward_output[
                        "weight_norm"
                    ]

                    # Add submodule gradient norms if available
                    if "submodule_norms" in backward_output:
                        for submodule_name, norms in backward_output[
                            "submodule_norms"
                        ].items():
                            metrics[f"Gradients/{submodule_name}_GradNorm"] = norms[
                                "grad_norm"
                            ]
                            metrics[f"Gradients/{submodule_name}_WeightNorm"] = norms[
                                "weight_norm"
                            ]

                # Compute evaluation metrics (vectorized over batch)
                eval_metrics = engine_helpers.compute_eval_metrics(
                    pred_decomposition,
                    gt_decomposition,
                    phase,
                    pixel_supervision_mask,
                    pixel_inpaint_mask,
                )
                metrics.update(eval_metrics)

                # Update the metrics dataframe
                self.metrics[phase] = pd.concat(
                    [self.metrics[phase], pd.DataFrame(metrics, index=[0])],
                    ignore_index=True,
                )

                # Image logging to wandb - with aggressive cleanup after
                if log_images_this_batch and self.wandb:
                    gt_data = {
                        "rgb": gt_decomposition["rgb_highlighted"].cpu(),
                    }
                    if "highlight" in gt_decomposition:
                        gt_data["highlight"] = gt_decomposition["highlight"].cpu()
                    if "rgb_highlighted" in gt_decomposition:
                        gt_data["rgb_highlighted"] = gt_decomposition[
                            "rgb_highlighted"
                        ].cpu()

                    image_size = self.config.MODEL.RGB_ENCODER.IMAGE_SIZE
                    engine_viz.prepare_decomposition_dicts_for_logging(
                        gt_decomposition,
                        pred_decomposition,
                        diffuse_teacher_tokens,
                        patch_inpaint_mask,
                        patch_supervision_mask,
                        pixel_supervision_mask,
                        diffuse,
                        image_size,
                    )

                    images = self.create_visualization_images(
                        gt_decomposition,
                        pred_decomposition,
                        gt_data,
                        as_single_panel=True,
                        also_save_individual_images=False,
                        batch_idx=batch_idx,
                        phase=phase,
                        test_idx=test_idx if phase == "Test" else None,
                    )
                    if images:
                        images_logged = True

                # Console logging (rank 0 only when DDP)
                if (
                    self._is_main_process()
                    and batch_idx % self.config.get("LOG_INTERVAL", 10) == 0
                ):
                    extra_info = (
                        "W" if is_training and step < self.warmup_steps else None
                    )
                    self.console_log_metrics(
                        stage=phase,
                        epoch=self.step["epoch"],
                        batch_idx=batch_idx,
                        dataloader_len=len(dataloader),
                        extra_info=extra_info,
                    )

                # WandB logging the batch metrics (rank 0 only when DDP)
                if (
                    self._is_main_process()
                    and self.wandb
                    and batch_idx % self.logfreq_wandb == 0
                ):
                    # Use the self._prepare_metrics_for_wandb function to format metrics properly
                    wandb_metrics = self._prepare_metrics_for_wandb(metrics, phase)
                    # Add the batch number
                    if phase == "Training":
                        batch_str = "batch"
                    elif phase == "Validation":
                        batch_str = "valbatch"
                    elif phase == "Test":
                        batch_str = f"test_idx_{test_idx}"
                    wandb_metrics[f"Step/{batch_str}"] = self.step[f"{phase}_batch"]
                    if phase == "Test" and test_idx is not None:
                        wandb_metrics["Step/test_idx"] = test_idx
                        # Rewrite all Test metric keys to include test index as prefix
                        prefixed_metrics = {}
                        for key, value in wandb_metrics.items():
                            if isinstance(key, str) and key.startswith("Test/"):
                                rest = key[len("Test/") :]
                                new_key = f"Test/test_idx_{test_idx}/{rest}"
                                prefixed_metrics[new_key] = value
                            else:
                                prefixed_metrics[key] = value
                        wandb_metrics = prefixed_metrics
                    self.wandb.log(wandb_metrics)

                # Strategic memory cleanup - more aggressive for larger batches
                self._strategic_memory_cleanup(
                    phase, batch_idx, exclude_vars=["self", "dataloader"]
                )

                # Clean up variables that are no longer needed
                del gt_decomposition, pred_decomposition, losses, loss_value
                if backward_output is not None:
                    del backward_output

                # Force cleanup every few batches for maximum memory efficiency
                if batch_idx % self.memory_cleanup_frequency == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                # Emergency save checkpoint if time left in job is less than 2 minutes
                if self._is_main_process() :
                    time_left = get_slurm_time_left_minutes()
                    if (time_left is not None) and (time_left < 2):
                        save_filename = f"emergency_chkpt_E{self.step['epoch'] + 1}B{batch_idx}.pt"
                        emergency_cehckpoint_exists = glob.glob(os.path.join(self.earlystopping.checkpointpath, "emergency_chkpt*"))
                        if not emergency_cehckpoint_exists:
                            self.logger.info(
                                f"Less than 2 minutes left in job execution time. Emergency saving checkpoint at {save_filename}",
                                context="SAVE",
                            )
                            self.earlystopping.save_checkpoint(
                                val_loss=0.0,
                                model=self.model,
                                epoch=self.step["epoch"],
                                config=self.config,
                                filename_override=save_filename,
                            )

        # Compute average loss for epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        if self._is_ddp and phase in ("Training", "Validation"):
            # All-reduce so all ranks have the same avg_loss for early stopping / scheduler
            loss_sum = torch.tensor(
                [sum(epoch_losses), len(epoch_losses)],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            total_loss, total_count = loss_sum[0].item(), loss_sum[1].item()
            avg_loss = total_loss / total_count if total_count > 0 else 0.0
        if self._is_main_process():
            self.logger.info(
                f"Epoch {self.step['epoch'] + 1} - Average Loss: {avg_loss:.6f}",
                context=phase.upper(),
            )

        # Log epoch metrics to wandb (rank 0 only when DDP)
        if self._is_main_process() and self.wandb:
            # For Test phase, filter using the stable column 'Step/test_idx'.
            # For Training/Validation, filter using 'Step/epoch' as before.
            if phase == "Test" and test_idx is not None:
                df = self.metrics[phase]
                if "Step/test_idx" in df.columns:
                    selector = df["Step/test_idx"] == test_idx
                    df_mean = df[selector].mean(numeric_only=True)
                else:
                    # Fallback: no explicit test_idx column found; average all rows
                    df_mean = df.mean(numeric_only=True)
                epoch_label = f"test_idx_{test_idx}"
            else:
                df_mean = self.metrics[phase][
                    self.metrics[phase]["Step/epoch"] == self.step["epoch"]
                ].mean()
                epoch_label = "epoch"

            epoch_metrics = self._prepare_metrics_for_wandb(df_mean, phase)
            # Build final logging dict with strict Test prefixing convention
            final_epoch_metrics = {}
            if phase == "Test" and test_idx is not None:
                for key, value in epoch_metrics.items():
                    if key.startswith("Test/"):
                        rest = key[len("Test/") :]
                        new_key = f"Test/test_idx_{test_idx}/{rest}"
                        final_epoch_metrics[new_key] = value
                # Record step keys
                final_epoch_metrics[f"Step/test_idx_{test_idx}"] = test_idx
                final_epoch_metrics["Step/test_idx"] = test_idx
            else:
                # Training/Validation: keep standard epoch prefixing
                for key, value in epoch_metrics.items():
                    if key.startswith(phase + "/"):
                        final_epoch_metrics[
                            key.replace(phase, f"{phase}/{epoch_label}")
                        ] = value
                final_epoch_metrics[f"Step/{epoch_label}"] = self.step["epoch"]

            self.wandb.log(final_epoch_metrics)

        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with enhanced state information. Only rank 0 when DDP."""
        if not self._is_main_process():
            return
        # Unwrap DataParallel and DDP so we save the real model state_dict
        m = getattr(self.model, "module", self.model)
        model_for_save = getattr(m, "module", m)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_for_save.state_dict(),
            "model_class_name": model_for_save.__class__.__name__,
            "model_class_module": model_for_save.__class__.__module__,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "runname": getattr(self, "runname", None),
            "wandb_run_id": getattr(self.wandb, "id", None) if self.wandb else None,
        }

        # Add scheduler states if available (save what Engine actually uses)
        if hasattr(self, "LRscheduler") and self.LRscheduler is not None:
            try:
                checkpoint["LRscheduler_state_dict"] = self.LRscheduler.state_dict()
            except Exception as e:
                self.logger.warning(f"Could not save LRscheduler state: {e}")
        if hasattr(self, "LRschedulerPlateau") and self.LRschedulerPlateau is not None:
            try:
                checkpoint["LRschedulerPlateau_state_dict"] = (
                    self.LRschedulerPlateau.state_dict()
                )
            except Exception as e:
                self.logger.warning(f"Could not save LRschedulerPlateau state: {e}")

        # Add early stopping state if available
        if hasattr(self, "earlystopping") and self.earlystopping is not None:
            checkpoint["earlystopping_state"] = {
                "val_loss_min": getattr(
                    self.earlystopping, "val_loss_min", float("inf")
                ),
                "counter": getattr(self.earlystopping, "counter", 0),
                "patience": getattr(self.earlystopping, "patience", 0),
            }

        # Add training metrics history if available
        if hasattr(self, "training_metrics") and self.training_metrics:
            checkpoint["training_metrics_history"] = self.training_metrics
        if hasattr(self, "validation_metrics") and self.validation_metrics:
            checkpoint["validation_metrics_history"] = self.validation_metrics

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.MODELS_DIR, f"checkpoint_epoch_{epoch + 1}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.MODELS_DIR, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch + 1}", context="SAVE")

    @contextmanager
    def choose_if_grad(self, mode):
        """Conditionally use torch.no_grad based on the given mode."""
        with torch.no_grad() if mode in ["Validation", "Test"] else nullcontext():
            yield

    def create_visualization_images(
        self,
        gt_decomposition,
        pred_decomposition,
        sample,
        as_single_panel=True,
        batch_idx=0,
        phase: str = None,
        test_idx: Optional[int] = None,
        also_save_individual_images: bool = False,
    ):
        """
        Creates visualization images for polarization-based reflection removal training.

        Args:
            gt_decomposition (dict): Ground truth decomposition (rgb, AoP, DoP, etc.)
            pred_decomposition (dict): Model output (specular, diffuse, recon, etc.)
            sample (dict): Original sample from dataset
            as_single_panel (bool): Whether to create a single comparison panel
            batch_idx (int): Batch index to visualize
            phase (str): Phase name (Training/Validation/Test)
            test_idx (Optional[int]): Test index for Test phase
            also_save_individual_images (bool): Also save key images separately

        Returns:
            dict: Dictionary of wandb.Image objects for visualization
        """

        def add_image_fn(viz_dict, key, tensor, caption, bi, ph, ti):
            self._add_image_safely(viz_dict, key, tensor, caption, bi, ph, ti)

        return engine_viz.create_visualization_images(
            gt_decomposition,
            pred_decomposition,
            sample,
            add_image_fn,
            as_single_panel=as_single_panel,
            batch_idx=batch_idx,
            phase=phase,
            test_idx=test_idx,
            also_save_individual_images=also_save_individual_images,
        )

    def reinstantiate_model_from_checkpoint(self, checkpoint_path=None):
        """
        Reinstantiate the model from checkpoint.
        """
        if checkpoint_path is None:
            # Try to load best model
            checkpoint_path = os.path.join(self.MODELS_DIR, "full_model_weights.pt")

        if not os.path.exists(checkpoint_path):
            self.logger.warning(
                f"Checkpoint not found at {checkpoint_path}", context="SAVE"
            )
            return

        try:
            if not os.path.exists(checkpoint_path):
                # Try swapping .pth <-> .pt if file not found
                if checkpoint_path.endswith(".pt"):
                    alt_checkpoint_path = checkpoint_path[:-3] + ".pth"
                elif checkpoint_path.endswith(".pth"):
                    alt_checkpoint_path = checkpoint_path[:-4] + ".pt"
                else:
                    alt_checkpoint_path = None

                if alt_checkpoint_path and os.path.exists(alt_checkpoint_path):
                    checkpoint_path = alt_checkpoint_path
                else:
                    return None

            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            state_dict = checkpoint["model_state_dict"]
            # Strip "module." prefix if checkpoint was saved with DataParallel
            state_dict = {
                k.replace("module.", "", 1) if k.startswith("module.") else k: v
                for k, v in state_dict.items()
            }
            # Load into unwrapped real model (DataParallel and/or DataParallelWrapper)
            load_target = getattr(
                getattr(self.model, "module", self.model),
                "module",
                getattr(self.model, "module", self.model),
            )
            load_target.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Load schedulers if present (support both new and legacy keys)
            try:
                if (
                    "LRscheduler_state_dict" in checkpoint
                    and hasattr(self, "LRscheduler")
                    and self.LRscheduler is not None
                ):
                    self.LRscheduler.load_state_dict(
                        checkpoint["LRscheduler_state_dict"]
                    )
                if (
                    "LRschedulerPlateau_state_dict" in checkpoint
                    and hasattr(self, "LRschedulerPlateau")
                    and self.LRschedulerPlateau is not None
                ):
                    self.LRschedulerPlateau.load_state_dict(
                        checkpoint["LRschedulerPlateau_state_dict"]
                    )
                # Backward-compatibility: legacy single scheduler key
                if "scheduler_state_dict" in checkpoint:
                    loaded = False
                    if hasattr(self, "LRscheduler") and self.LRscheduler is not None:
                        try:
                            self.LRscheduler.load_state_dict(
                                checkpoint["scheduler_state_dict"]
                            )
                            loaded = True
                        except Exception:
                            pass
                    if (
                        not loaded
                        and hasattr(self, "LRschedulerPlateau")
                        and self.LRschedulerPlateau is not None
                    ):
                        try:
                            self.LRschedulerPlateau.load_state_dict(
                                checkpoint["scheduler_state_dict"]
                            )
                            loaded = True
                        except Exception:
                            pass
                    if not loaded:
                        self.logger.warning(
                            "Found legacy scheduler_state_dict but no compatible scheduler attribute to load into",
                            context="SAVE",
                        )
            except Exception as e:
                self.logger.warning(f"Could not load scheduler state(s): {e}")
            self.logger.info(
                f"Model reinstantiated from checkpoint: {checkpoint_path}",
                context="SAVE",
            )
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}", context="SAVE")

    @staticmethod
    def create_model_from_checkpoint(checkpoint_path, device=None):
        """
        Create a model instance from checkpoint information.

        Args:
            checkpoint_path (str): Path to the checkpoint file
            device (torch.device, optional): Device to load the model on

        Returns:
            dict with keys model, optimizer_state_dict, epoch, config, etc., or None if loading fails
        """
        return engine_helpers.create_model_from_checkpoint(checkpoint_path, device)

    def _reapply_config_lrs_on_resume(self):
        """Re-apply per-module learning rates from the CURRENT config after a resume has
        restored the optimizer + scheduler state.

        On --resume-run the engine restores ``optimizer.load_state_dict`` and the cosine /
        plateau scheduler states, which silently overrides any LR edits made in the YAML
        (``DECODER_LR``, ``TOKEN_INPAINTER_LR``, ``RGB_ENCODER_LR``, ``LR_SCHEDULER``). When
        the config flag ``REAPPLY_LRS_ON_RESUME`` is set, this overwrites the restored
        per-group LRs with the config values and rebuilds the schedulers from scratch, so the
        new LRs become the starting point of a fresh anneal over the remaining epochs.

        The optimizer param-group *structure* is never changed (components frozen via LR=0.0
        must stay frozen), so the preceding ``optimizer.load_state_dict`` still succeeds.
        """
        config = self.config

        # Resolve per-component config LRs exactly as initialize.optimizers does.
        encoder_lr = config.MODEL.RGB_ENCODER.RGB_ENCODER_LR
        token_lr = config.MODEL.TOKEN_INPAINTER.TOKEN_INPAINTER_LR
        decoders_config = config.MODEL.DECODERS
        decoder_lrs = {
            name: decoders_config[name].DECODER_LR for name in decoders_config.keys()
        }

        # Unwrap DataParallel / DDP / DataParallelWrapper so names match the real model.
        from utilities.model import DataParallelWrapper

        effective_model = self.model
        if isinstance(
            effective_model,
            (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
        ):
            effective_model = effective_model.module
        if isinstance(effective_model, DataParallelWrapper):
            effective_model = effective_model._modules.get("module", effective_model)

        # Map each parameter tensor (by id) to its config LR via the same bucketing as the
        # optimizer builder, so we can set the right LR on each (per-component) param group.
        id_to_lr = {}
        for name, p in effective_model.named_parameters():
            if name.startswith("dinov3."):
                id_to_lr[id(p)] = encoder_lr
            elif name.startswith("token_inpaint."):
                id_to_lr[id(p)] = token_lr
            elif name.startswith("decoders."):
                parts = name.split(".")
                if len(parts) >= 2:
                    id_to_lr[id(p)] = decoder_lrs.get(parts[1])

        # Overwrite each param group's lr AND initial_lr (so freshly-built schedulers adopt
        # the new value as base_lr; CosineAnnealingLR uses setdefault on initial_lr).
        new_group_lrs = []
        for group in self.optimizer.param_groups:
            group_lrs = {id_to_lr[id(p)] for p in group["params"] if id(p) in id_to_lr}
            group_lrs.discard(None)
            if len(group_lrs) == 1:
                lr = group_lrs.pop()
            elif group_lrs:
                lr = max(group_lrs)  # mixed group (unexpected) — pick the largest
            else:
                lr = group["lr"]  # unknown component — leave untouched
            group["lr"] = lr
            group["initial_lr"] = lr
            new_group_lrs.append(lr)

        # Rebuild schedulers from config: cosine base_lrs == the new LRs and the anneal
        # restarts from the top over config.EPOCHS, discarding the restored scheduler states.
        sched = initialize.schedulers(self.optimizer, config, self.training_dl)
        self.LRscheduler = sched["LRscheduler"]
        self.LRschedulerPlateau = sched["LRschedulerPlateau"]

        self.logger.info(
            "Re-applied config LRs on resume (REAPPLY_LRS_ON_RESUME=True): "
            + ", ".join(f"{lr:.2e}" for lr in new_group_lrs)
            + " — schedulers rebuilt from config.",
            context="RESUME",
        )
        return True

    def resume_from_run(self, run_identifier: str) -> bool:
        """
        Resume training from an existing run.

        Args:
            run_identifier (str): Run name or run ID to resume from

        Returns:
            bool: True if resume was successful, False otherwise
        """
        from utilities.run_resume import get_resume_info, load_checkpoint_for_resume

        # Get resume information
        self.logger.markup = False
        resume_info = get_resume_info(run_identifier, self.RUNS_DIR)
        if resume_info is None:
            self.logger.error(f"Failed to get resume info for run: {run_identifier}")
            return False

        # Set up directories using the existing run information
        self._setup_resume_directories(resume_info)

        # Load the latest checkpoint
        checkpoint_data = load_checkpoint_for_resume(
            resume_info["latest_checkpoint"], self.device
        )
        if checkpoint_data is None:
            self.logger.error("Failed to load checkpoint data")
            return False

        # Load model state into unwrapped real model (DataParallel and/or DataParallelWrapper)
        try:
            state_dict = checkpoint_data["model_state_dict"]
            state_dict = {
                k.replace("module.", "", 1) if k.startswith("module.") else k: v
                for k, v in state_dict.items()
            }
            load_target = getattr(
                getattr(self.model, "module", self.model),
                "module",
                getattr(self.model, "module", self.model),
            )
            load_target.load_state_dict(state_dict)
            self.logger.info("Loaded model state from checkpoint", context="RESUME")
        except Exception as e:
            self.logger.error(f"Failed to load model state: {e}")
            return False

        # Load optimizer state
        try:
            self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            self.logger.info("Loaded optimizer state from checkpoint", context="RESUME")
        except Exception as e:
            self.logger.error(f"Failed to load optimizer state: {e}")
            return False

        # Load scheduler states if available (support both new and legacy keys)
        try:
            if (
                "LRscheduler_state_dict" in checkpoint_data
                and hasattr(self, "LRscheduler")
                and self.LRscheduler is not None
            ):
                self.LRscheduler.load_state_dict(
                    checkpoint_data["LRscheduler_state_dict"]
                )
                self.logger.info(
                    "Loaded LRscheduler state from checkpoint", context="RESUME"
                )
            if (
                "LRschedulerPlateau_state_dict" in checkpoint_data
                and hasattr(self, "LRschedulerPlateau")
                and self.LRschedulerPlateau is not None
            ):
                self.LRschedulerPlateau.load_state_dict(
                    checkpoint_data["LRschedulerPlateau_state_dict"]
                )
                self.logger.info(
                    "Loaded LRschedulerPlateau state from checkpoint", context="RESUME"
                )
            # Backward-compatibility: legacy single scheduler key
            if "scheduler_state_dict" in checkpoint_data:
                loaded = False
                if hasattr(self, "LRscheduler") and self.LRscheduler is not None:
                    try:
                        self.LRscheduler.load_state_dict(
                            checkpoint_data["scheduler_state_dict"]
                        )
                        loaded = True
                        self.logger.info(
                            "Loaded legacy scheduler state into LRscheduler",
                            context="RESUME",
                        )
                    except Exception:
                        pass
                if (
                    not loaded
                    and hasattr(self, "LRschedulerPlateau")
                    and self.LRschedulerPlateau is not None
                ):
                    try:
                        self.LRschedulerPlateau.load_state_dict(
                            checkpoint_data["scheduler_state_dict"]
                        )
                        loaded = True
                        self.logger.info(
                            "Loaded legacy scheduler state into LRschedulerPlateau",
                            context="RESUME",
                        )
                    except Exception:
                        pass
                if not loaded:
                    self.logger.warning(
                        "Found legacy scheduler_state_dict but no compatible scheduler attribute to load into",
                        context="RESUME",
                    )
        except Exception as e:
            self.logger.warning(f"Failed to load scheduler state(s): {e}")

        # Optionally re-apply config LRs over the just-restored optimizer/scheduler state so
        # YAML LR edits take effect on resume (opt-in via REAPPLY_LRS_ON_RESUME).
        if self.config.get("REAPPLY_LRS_ON_RESUME", False):
            try:
                self._reapply_config_lrs_on_resume()
            except Exception as e:
                self.logger.warning(
                    f"Failed to re-apply config LRs on resume: {e}", context="RESUME"
                )

        # Restore early stopping state if available
        if (
            "earlystopping_state" in checkpoint_data
            and hasattr(self, "earlystopping")
            and self.earlystopping is not None
        ):
            try:
                es_state = checkpoint_data["earlystopping_state"]
                self.earlystopping.val_loss_min = es_state.get(
                    "val_loss_min", float("inf")
                )
                self.earlystopping.counter = es_state.get("counter", 0)
                self.earlystopping.patience = es_state.get("patience", 0)
                self.logger.info("Restored early stopping state", context="RESUME")
            except Exception as e:
                self.logger.warning(f"Failed to restore early stopping state: {e}")

        # Restore metrics history if available
        if "training_metrics_history" in checkpoint_data:
            self.training_metrics = checkpoint_data["training_metrics_history"]
            self.logger.info("Restored training metrics history", context="RESUME")

        if "validation_metrics_history" in checkpoint_data:
            self.validation_metrics = checkpoint_data["validation_metrics_history"]
            self.logger.info("Restored validation metrics history", context="RESUME")

        # Set the starting epoch (checkpoint["epoch"] is 0-based last completed epoch)
        last_completed_epoch = int(checkpoint_data["epoch"])
        self.start_epoch = last_completed_epoch + 1
        self.epochs = self.config["EPOCHS"] + self.start_epoch
        self.step["epoch"] = self.start_epoch
        self.logger.info(
            f"Resuming training from epoch {self.start_epoch + 1} (1-based; next epoch index {self.start_epoch})",
            context="RESUME",
        )
        self.logger.info(
            f"Epoch range for this run: {self.start_epoch} to {self.epochs - 1} (total {self.epochs - self.start_epoch} epochs to run)",
            context="RESUME",
        )

        # Log resumed-from epoch to WandB so the run summary shows correct continuity
        if self._is_main_process() and self.wandb is not None:
            self.wandb.summary["resumed_from_epoch_index"] = last_completed_epoch
            self.wandb.summary["next_epoch_index"] = self.start_epoch

        # Update wandb run ID if available and re-initialize wandb only if needed.
        if "wandb_run_id" in checkpoint_data and checkpoint_data["wandb_run_id"]:
            target_run_id = checkpoint_data["wandb_run_id"]
            current_run_id = (
                getattr(self.wandb, "id", None)
                if getattr(self, "wandb", None)
                else None
            )

            # Only reinitialize WandB if we are not already attached to the correct run.
            if target_run_id != current_run_id:
                self.resume_wandb_run_id = target_run_id
                self.logger.info(
                    f"Will resume wandb run: {self.resume_wandb_run_id}",
                    context="RESUME",
                )

                # Re-initialize wandb with the correct run ID
                self._reinitialize_wandb()

        return True

    def _setup_resume_directories(self, resume_info):
        """Set up directories using existing run information"""
        # Extract run name from the run directory path
        run_dir = resume_info["run_dir"]
        runname = os.path.basename(run_dir)

        # Set up directory structure using existing run
        self.runname = runname
        self.RUN_DIR = run_dir
        self.MODELS_DIR = resume_info["models_dir"]
        self.TEST_DIR = os.path.join(run_dir, "tests")
        self.paths_file = os.path.join(run_dir, "loadeddata.csv")

        # Create test directory if it doesn't exist
        os.makedirs(self.TEST_DIR, exist_ok=True)

        # Initialize early stopping with the existing run name
        from utilities import engine_initializers as initialize

        earlystopping_result = initialize.earlystopping(
            self.earlystopping_patience,
            self.MODELS_DIR,
            self.runname,
        )

        # Set early stopping attributes
        for key, value in earlystopping_result.items():
            setattr(self, key, value)

        # Update config with the run name
        self.config["name"] = self.runname

        self.logger.info(
            f"Set up resume directories for run: {self.runname}", context="RESUME"
        )
        self.logger.info(f"Run directory: {self.RUN_DIR}", context="RESUME")

    def _reinitialize_wandb(self):
        """Re-initialize wandb with the correct run ID for resuming"""
        if self.wandb is not None:
            # Finish the current wandb run
            self.wandb.finish()

        # Re-initialize wandb with the resume run ID
        from utilities import engine_initializers as initialize

        wandb_result = initialize.wandb(
            self.config,
            self.model,
            self.config.get("NOTES", ""),
            False,  # no_wandb
            self.resume_wandb_run_id,
        )

        # Update the wandb reference
        self.wandb = wandb_result.get("wandb")
        self.logger.info(
            f"Re-initialized wandb with run ID: {self.resume_wandb_run_id}",
            context="RESUME",
        )

    def _log_memory_usage(self, context: str = ""):
        """Log current GPU memory usage for monitoring."""
        engine_memory.log_memory_usage(self.logger, context, self.memory_monitoring)

    def _aggressive_memory_cleanup(self, exclude_vars: list = None):
        """Perform aggressive memory cleanup using the gpuClean utility."""
        engine_memory.aggressive_memory_cleanup(
            self.logger,
            self.aggressive_cleanup,
            self.memory_monitoring,
            exclude_vars,
        )

    def _strategic_memory_cleanup(
        self, phase: str, batch_idx: int, exclude_vars: list = None
    ):
        """Perform strategic memory cleanup based on phase and batch index."""
        engine_memory.strategic_memory_cleanup(
            phase,
            batch_idx,
            self.memory_cleanup_frequency,
            self.logger,
            self.aggressive_cleanup,
            self.memory_monitoring,
            exclude_vars,
        )

    def _cleanup_tensor_dict(self, tensor_dict: dict, keys_to_keep: list = None):
        """Clean up a dictionary of tensors, optionally keeping specified keys."""
        engine_helpers.cleanup_tensor_dict(tensor_dict, keys_to_keep)

    def _prepare_metrics_for_wandb(self, metrics, phase):
        """Format metrics with phase prefix for wandb."""
        return engine_helpers.prepare_metrics_for_wandb(metrics, phase)

    def _to_cpu_image(self, tensor, batch_idx=0):
        """Convert tensor to PIL Image with memory optimization."""
        return engine_helpers.to_cpu_image(tensor, batch_idx)

    def _add_image_safely(
        self,
        viz_dict,
        key,
        tensor,
        caption,
        batch_idx=0,
        phase: str = None,
        test_idx: Optional[int] = None,
    ):
        """Add image to viz dict and log to wandb with phase/test_idx prefixing."""
        engine_helpers.add_image_safely(
            viz_dict,
            key,
            tensor,
            caption,
            batch_idx,
            phase,
            test_idx,
            self.metrics["Test"] if "Test" in self.metrics else None,
        )

    def _load_and_increment_test_index(self) -> int:
        """Load current test index from RUN_DIR and increment it for next test.
        Returns the current index to be used for this test.
        When DDP, only rank 0 reads/writes the file.
        """
        if self._is_ddp and not self._is_main_process():
            return 0
        idx_path = os.path.join(self.RUN_DIR, "test_index.txt")
        current = 0
        try:
            if os.path.exists(idx_path):
                with open(idx_path, "r") as f:
                    content = f.read().strip()
                    if content != "":
                        current = int(content)
        except Exception:
            current = 0
        # Increment and persist for the next test
        try:
            with open(idx_path, "w") as f:
                f.write(str(current + 1))
        except Exception:
            pass
        return current

    # except Exception as e:
    #     self.logger.warning(f"Failed to create {key} visualization: {e}")
