# Standard Library
import logging
import pathlib
import typing

# Third Party
import torch
import torch.nn as nn

# Project
from pretraining.configs.training import checkpointer_configs

logger = logging.getLogger(__name__)


class Checkpointer:
    """
    Handles saving and loading training checkpoints.

    Checkpoint format:
    {
        'model': model_state_dict,
        'optimizer': optimizer_state_dict,
        'model_args': dict of model constructor args,
        'iter_num': current iteration,
        'best_val_loss': best validation loss seen,
        'config': full training config
    }
    """

    def __init__(self, config: checkpointer_configs.CheckpointerConfig) -> None:
        self.config = config
        self.save_dir = pathlib.Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.save_dir / "ckpt.pt"

        # Track best validation loss for save_best functionality
        self.best_val_loss = float("inf")

        # Track saved checkpoints for rotation (keep_last_n)
        self.saved_checkpoints: typing.List[pathlib.Path] = []

    def should_save_checkpoint(self, iter_num: int) -> bool:
        """Check if we should save a checkpoint at this iteration."""
        return iter_num % self.config.save_interval == 0

    def should_save_best(self, current_val_loss: float) -> bool:
        """Check if this is the best model so far."""
        if not self.config.save_best:
            return False

        is_best = current_val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = current_val_loss
        return is_best

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iter_num: int,
        best_val_loss: float,
        model_args: typing.Dict[str, typing.Any],
        config: typing.Dict[str, typing.Any],
    ) -> None:
        """
        Save a training checkpoint.

        Args:
            model: Model to save (will handle DDP unwrapping)
            optimizer: Optimizer state to save
            iter_num: Current iteration number
            best_val_loss: Best validation loss achieved
            model_args: Model constructor arguments
            config: Full configuration dict
        """
        # Unwrap DDP if needed
        raw_model = model.module if hasattr(model, "module") else model

        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": model_args,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "config": config,
        }

        # Save regular checkpoint
        checkpoint_name = f"ckpt_iter{iter_num}.pt"
        checkpoint_path = self.save_dir / checkpoint_name

        logger.info(f"Saving checkpoint to {checkpoint_path}")

        # Save to temporary file first, then rename (atomic operation)
        temp_path = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)

        # Update symlink to latest checkpoint
        latest_link = self.save_dir / "ckpt_latest.pt"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_name)

        # Track saved checkpoint
        self.saved_checkpoints.append(checkpoint_path)

        # Rotate old checkpoints if needed
        self._rotate_checkpoints()

        # Save best model if this is the best so far
        if self.should_save_best(best_val_loss):
            best_path = self.save_dir / "ckpt_best.pt"
            logger.info(f"New best model! Saving to {best_path}")
            torch.save(checkpoint, best_path)

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to keep only the last N."""
        if self.config.keep_last_n <= 0:
            return  # Keep all checkpoints

        while len(self.saved_checkpoints) > self.config.keep_last_n:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                logger.info(f"Removing old checkpoint: {old_checkpoint}")
                old_checkpoint.unlink()

    def find_resume_checkpoint(self) -> typing.Optional[pathlib.Path]:
        """Find checkpoint to resume from based on config."""
        if self.config.resume_from:
            # Explicit path provided
            resume_path = pathlib.Path(self.config.resume_from)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            return resume_path

        # Try to find latest checkpoint
        latest_link = self.save_dir / "ckpt_latest.pt"
        if latest_link.exists():
            return latest_link

        # No checkpoint to resume from
        return None

    def load_checkpoint(self, device: str) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """
        Load a training checkpoint for resuming.

        Args:
            device: Device to load tensors to

        Returns:
            Dictionary containing checkpoint data, or None if no checkpoint found
        """
        checkpoint_path = self.find_resume_checkpoint()
        if checkpoint_path is None:
            logger.info("No checkpoint found to resume from")
            return None

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Update best val loss if resuming
        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]

        return checkpoint

    def checkpoint_exists(self) -> bool:
        """Check if a checkpoint exists for resuming."""
        return self.find_resume_checkpoint() is not None

    def save_final_model(self, model: nn.Module, iter_num: int, val_loss: float) -> None:
        """
        Save just the final model weights (no optimizer state).
        Useful for deployment/inference.
        """
        # Unwrap DDP if needed
        raw_model = model.module if hasattr(model, "module") else model

        final_path = self.save_dir / f"model_iter{iter_num}_loss{val_loss:.4f}.pt"

        model_state = {
            "model": raw_model.state_dict(),
            "config": raw_model.config,
            "iter_num": iter_num,
            "val_loss": val_loss,
        }

        logger.info(f"Saving final model to {final_path}")
        torch.save(model_state, final_path)
