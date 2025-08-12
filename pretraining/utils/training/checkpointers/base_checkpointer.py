# Standard Library
import abc
import pathlib
import typing

# Third Party
import torch

# Project
from pretraining.configs import core
from pretraining.configs.training import checkpointer_configs
from pretraining.trainer import checkpoint_data


class BaseCheckpointer(abc.ABC):
    """Abstract base class for all checkpointer implementations.

    Defines the interface that all checkpointers must implement,
    whether they handle regular checkpoints, sharded FSDP checkpoints,
    or other checkpoint formats.
    """

    def __init__(self, config: checkpointer_configs.CheckpointerConfig) -> None:
        """Initialize the base checkpointer.

        Args:
            config: Checkpointer configuration
        """
        self.config = config
        self.save_dir = pathlib.Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Track best validation loss for save_best functionality
        self.best_val_loss = float("inf")

    def should_save_checkpoint(self, iter_num: int) -> bool:
        """Check if we should save a checkpoint at this iteration.

        Args:
            iter_num: Current iteration number

        Returns:
            True if checkpoint should be saved
        """
        return iter_num % self.config.save_interval == 0

    def should_save_best(self, current_val_loss: float) -> bool:
        """Check if this is the best model so far.

        Args:
            current_val_loss: Current validation loss

        Returns:
            True if this is the best model and should be saved
        """
        if not self.config.save_best:
            return False

        is_best = current_val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = current_val_loss
        return is_best

    @abc.abstractmethod
    def save_checkpoint(
        self, ckpt_data: checkpoint_data.CheckpointData, config: core.TrainerConfig
    ) -> None:
        """Save a training checkpoint.

        Args:
            ckpt_data: CheckpointData object containing all state to save
            config: Full training configuration to save separately
        """
        pass

    @abc.abstractmethod
    def load_checkpoint(self, device: str) -> typing.Optional[checkpoint_data.CheckpointData]:
        """Load a training checkpoint for resuming.

        Args:
            device: Device to load tensors to

        Returns:
            CheckpointData object, or None if no checkpoint found
        """
        pass

    @abc.abstractmethod
    def find_resume_checkpoint(self) -> typing.Optional[pathlib.Path]:
        """Find checkpoint to resume from based on config.

        Returns:
            Path to checkpoint file, or None if not found
        """
        pass

    def checkpoint_exists(self) -> bool:
        """Check if a checkpoint exists for resuming.

        Returns:
            True if a resumable checkpoint exists
        """
        return self.find_resume_checkpoint() is not None

    @abc.abstractmethod
    def save_model_only(
        self,
        model_state_dict: typing.Dict[str, torch.Tensor],
        metadata: typing.Dict[str, typing.Any],
    ) -> None:
        """Save just the model weights (no optimizer state).

        Useful for deployment/inference.

        Args:
            model_state_dict: Model state dictionary
            metadata: Additional metadata (config, iteration, loss, etc.)
        """
        pass
