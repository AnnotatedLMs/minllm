# Standard Library
import typing

# Third Party
import torch
from packaging import version

# Project
from pretraining.configs import core
from pretraining.configs.training import execution_configs
from pretraining.trainer import llm_trainer
from pretraining.utils import logger as logger_utils
from pretraining.utils.training import dist_utils
from pretraining.utils.training import model_wrapper
from pretraining.utils.training.checkpointers import checkpointer_factory

log = logger_utils.get_logger(__name__)


def prepare_model_for_training(
    model: torch.nn.Module,
    config: core.TrainerConfig,
    device: torch.device,
) -> torch.nn.Module:
    """
    Prepare model for training: device placement, wrapping, and parameter initialization.

    This function handles the common model setup pattern:
    1. Move model to device (except for FSDP which handles internally)
    2. Wrap model with appropriate distributed strategy
    3. Initialize parameters when needed

    Args:
        model: The raw model to prepare
        config: Training configuration
        device: Device to use for training

    Returns:
        The wrapped model ready for training

    Note:
        - FSDP handles device placement internally
        - Parameter initialization follows OLMo's pattern:
          * DDP always needs manual reset_parameters()
          * FSDP needs it only for PyTorch >= 2.1.0
    """
    execution_strategy = config.training.execution.strategy

    # Step 1: Move model to device (FSDP handles this internally)
    if execution_strategy != execution_configs.ExecutionStrategy.FSDP:
        model = model.to(device)

    # Step 2: Wrap model based on execution strategy
    log.info(f"Wrapping model with {execution_strategy.value}...")
    dist_model = model_wrapper.wrap_model(
        model,
        config.training.execution,
        config.training.precision_dtype,
    )

    # Step 3: Initialize parameters if needed
    # When param_init_fn is None, FSDP will call reset_parameters() automatically
    # For DDP or when using PyTorch >= 2.1.0 with FSDP, we need to call it manually
    if execution_strategy == execution_configs.ExecutionStrategy.DDP:
        model.reset_parameters()
    elif execution_strategy == execution_configs.ExecutionStrategy.FSDP:
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            model.reset_parameters()

    return dist_model


def load_checkpoint_if_resuming(
    trainer: llm_trainer.LLMTrainer,
    config: core.TrainerConfig,
) -> typing.Tuple[int, int]:
    """
    Try to load checkpoint if resuming from a previous run.

    Args:
        trainer: The trainer instance with checkpoint loading capability
        config: Training configuration containing checkpoint settings

    Returns:
        Tuple of (start_index, epoch) - tokens seen and epoch number from checkpoint,
        or (0, 0) if starting fresh

    Note:
        This function logs success/failure of checkpoint loading and extracts
        training state (tokens_seen, epoch) for dataset resumption.
    """
    start_index = 0
    epoch = 0

    should_resume, resume_path = checkpointer_factory.should_resume_training(
        config.training.checkpoint
    )

    if should_resume:
        log.info(f"Attempting to resume from {resume_path}")
        if trainer.load_checkpoint():
            log.info(f"Successfully resumed from iteration {trainer.state.iteration}")
            start_index = trainer.state.tokens_seen
            epoch = trainer.state.epoch
        else:
            log.warning(f"Failed to load checkpoint from {resume_path}, starting from scratch")

    return start_index, epoch


def save_final_checkpoint(trainer: llm_trainer.LLMTrainer) -> None:
    """
    Save final checkpoint, avoiding duplicate saves.

    This function checks if we just saved at the current step to avoid
    redundant checkpoint writes.

    Args:
        trainer: The trainer instance with checkpoint saving capability

    Note:
        Only the main process logs checkpoint save messages to avoid
        duplicate output in distributed training.
    """
    # Save final checkpoint only if we haven't just saved at this step
    if trainer.last_checkpoint_step != trainer.state.iteration:
        if dist_utils.is_main_process():
            log.info("Saving final checkpoint...")
        checkpoint_path = trainer.save_checkpoint()
        if dist_utils.is_main_process():
            log.info(f"Final checkpoint saved to {checkpoint_path}")
    else:
        if dist_utils.is_main_process():
            log.info(f"Final checkpoint already saved at step {trainer.state.iteration}")
