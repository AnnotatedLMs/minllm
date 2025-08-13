# Standard Library
import logging
import pathlib
import shutil
import typing

# Third Party
import torch
import yaml

# Project
from pretraining.configs import core
from pretraining.configs.training import checkpointer_configs
from pretraining.trainer import checkpoint_data
from pretraining.utils.training.checkpointers import base_checkpointer

logger = logging.getLogger(__name__)


class Checkpointer(base_checkpointer.BaseCheckpointer):
    """Standard checkpointer for non-sharded training.

    Handles regular checkpoint saving and loading for single-GPU
    and DDP training. Saves complete model and optimizer states
    in a single file.
    """

    def __init__(self, config: checkpointer_configs.CheckpointerConfig) -> None:
        """Initialize the standard checkpointer.

        Args:
            config: Checkpointer configuration
        """
        super().__init__(config)

        # Track saved checkpoints for rotation (keep_last_n)
        self.saved_checkpoints: typing.List[pathlib.Path] = []

    def save_checkpoint(
        self, ckpt_data: checkpoint_data.CheckpointData, config: core.TrainerConfig
    ) -> None:
        """Save a training checkpoint as a directory with separate files.

        Directory structure:
        - step{N}/
          - config.yaml
          - model.pt
          - optim.pt
          - scheduler.pt
          - train.pt

        Args:
            ckpt_data: CheckpointData object containing all state to save
            config: Full training configuration to save separately
        """
        # Get iteration from training state
        iter_num = ckpt_data.training_state.get("iteration", 0)
        best_val_loss = ckpt_data.training_state.get("best_val_loss", float("inf"))

        # Create checkpoint directory
        checkpoint_name = f"step{iter_num}"
        checkpoint_dir = self.save_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        # Save each component separately
        # 1. Save config as YAML
        config_path = checkpoint_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False)

        # 2. Save model state
        model_path = checkpoint_dir / "model.pt"
        torch.save(ckpt_data.model_state, model_path)

        # 3. Save optimizer state
        optim_path = checkpoint_dir / "optim.pt"
        torch.save(ckpt_data.optimizer_state, optim_path)

        # 4. Save scheduler state
        scheduler_path = checkpoint_dir / "scheduler.pt"
        torch.save(ckpt_data.scheduler_state, scheduler_path)

        # 5. Save training state
        train_path = checkpoint_dir / "train.pt"
        torch.save(ckpt_data.training_state, train_path)

        # Update symlink to latest checkpoint
        latest_link = self.save_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_name)

        # Track saved checkpoint
        self.saved_checkpoints.append(checkpoint_dir)

        # Rotate old checkpoints if needed
        self._rotate_checkpoints()

        # Save best model if this is the best so far
        if self.should_save_best(best_val_loss):
            best_link = self.save_dir / "best"
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(checkpoint_name)
            logger.info(f"New best model! Linked to {checkpoint_name}")

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to keep only the last N."""
        if self.config.keep_last_n <= 0:
            return  # Keep all checkpoints

        while len(self.saved_checkpoints) > self.config.keep_last_n:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                logger.info(f"Removing old checkpoint: {old_checkpoint}")
                shutil.rmtree(old_checkpoint)

    def find_resume_checkpoint(self) -> typing.Optional[pathlib.Path]:
        """Find checkpoint directory to resume from based on config.

        Returns:
            Path to checkpoint directory, or None if not found
        """
        if self.config.resume_from:
            # Explicit path provided
            resume_path = pathlib.Path(self.config.resume_from)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            return resume_path

        # Try to find latest checkpoint via symlink
        latest_link = self.save_dir / "latest"
        if latest_link.exists():
            # Resolve symlink to actual directory
            return latest_link.resolve()

        # No checkpoint to resume from
        return None

    def load_checkpoint(self, device: str) -> typing.Optional[checkpoint_data.CheckpointData]:
        """Load a training checkpoint from directory structure.

        Args:
            device: Device to load tensors to

        Returns:
            CheckpointData object, or None if no checkpoint found
        """
        checkpoint_dir = self.find_resume_checkpoint()
        if checkpoint_dir is None:
            logger.info("No checkpoint found to resume from")
            return None

        logger.info(f"Loading checkpoint from {checkpoint_dir}")

        # Load each component from separate files
        # Use weights_only=False for backward compatibility with PyTorch 2.6+
        # This is safe since we only load our own checkpoints
        model_state = torch.load(
            checkpoint_dir / "model.pt", map_location=device, weights_only=False
        )
        optimizer_state = torch.load(
            checkpoint_dir / "optim.pt", map_location=device, weights_only=False
        )
        scheduler_state = torch.load(
            checkpoint_dir / "scheduler.pt", map_location=device, weights_only=False
        )
        training_state = torch.load(
            checkpoint_dir / "train.pt", map_location=device, weights_only=False
        )

        # Create CheckpointData
        ckpt_data = checkpoint_data.CheckpointData(
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            training_state=training_state,
        )

        # Update best val loss if resuming
        if "best_val_loss" in training_state:
            self.best_val_loss = training_state["best_val_loss"]

        return ckpt_data

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
        iter_num = metadata.get("iteration", 0)
        val_loss = metadata.get("val_loss", 0.0)

        final_path = self.save_dir / f"model_iter{iter_num}_loss{val_loss:.4f}.pt"

        model_state = {"model": model_state_dict, **metadata}

        logger.info(f"Saving final model to {final_path}")
        torch.save(model_state, final_path)
