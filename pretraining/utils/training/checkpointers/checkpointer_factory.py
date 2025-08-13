"""Factory functions for building checkpointers on-demand."""

# Standard Library
import pathlib
import typing

# Third Party
import torch
from torch.distributed import fsdp

# Project
from pretraining.configs.training import checkpointer_configs
from pretraining.configs.training import execution_configs
from pretraining.utils.training.checkpointers import base_checkpointer
from pretraining.utils.training.checkpointers import core_checkpointer
from pretraining.utils.training.checkpointers import fsdp_checkpointer


def build_checkpointer(
    config: checkpointer_configs.CheckpointerConfig,
    execution: execution_configs.ExecutionConfig,
    dist_model: typing.Optional[torch.nn.Module] = None,
    optimizer: typing.Optional[torch.optim.Optimizer] = None,
) -> base_checkpointer.BaseCheckpointer:
    """Build a checkpointer based on execution strategy.

    Args:
        config: Checkpointer configuration
        execution: Execution configuration (determines strategy)
        dist_model: Distributed model (required for FSDP)
        optimizer: Optimizer (required for FSDP)

    Returns:
        Appropriate checkpointer for the execution strategy
    """
    if execution.strategy == execution_configs.ExecutionStrategy.FSDP:
        if dist_model is None or optimizer is None:
            raise ValueError("FSDP checkpointer requires dist_model and optimizer")
        if not isinstance(dist_model, fsdp.FullyShardedDataParallel):
            raise ValueError("FSDP checkpointer requires FSDP-wrapped model")
        return fsdp_checkpointer.FSDPCheckpointer(config, dist_model, optimizer)
    else:
        # For SINGLE and DDP strategies, use the standard checkpointer
        return core_checkpointer.Checkpointer(config)


def should_resume_training(
    checkpoint_config: checkpointer_configs.CheckpointerConfig,
) -> typing.Tuple[bool, typing.Optional[str]]:
    """Check if training should resume from a checkpoint.

    Args:
        checkpoint_config: Checkpoint configuration

    Returns:
        Tuple of (should_resume, resume_path_or_none)
    """
    if checkpoint_config.resume_from:
        resume_path = pathlib.Path(checkpoint_config.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Checkpoint specified in resume_from does not exist: {resume_path}"
            )
        return True, str(resume_path)
    return False, None
