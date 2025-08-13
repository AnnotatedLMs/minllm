# Standard Library
import logging
import pathlib
import shutil
import typing

# Third Party
import torch
import yaml
from torch.distributed import fsdp

# Project
from pretraining.configs import core
from pretraining.configs.training import checkpointer_configs
from pretraining.trainer import checkpoint_data
from pretraining.utils.training import dist_utils
from pretraining.utils.training.checkpointers import base_checkpointer

logger = logging.getLogger(__name__)


class FSDPCheckpointer(base_checkpointer.BaseCheckpointer):
    """Checkpointer for FSDP (Fully Sharded Data Parallel) training.

    Handles sharded checkpoint saving and loading where each rank
    saves its own shard of the model and optimizer states. This
    enables efficient checkpointing for large models that don't
    fit in a single GPU's memory.
    """

    def __init__(
        self,
        config: checkpointer_configs.CheckpointerConfig,
        model: fsdp.FullyShardedDataParallel,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Initialize the FSDP checkpointer.

        Args:
            config: Checkpointer configuration
            model: FSDP-wrapped model (needed for state dict configuration)
            optimizer: Optimizer (needed for sharded state dict extraction)
        """
        super().__init__(config)
        self.model = model
        self.optimizer = optimizer

        # Create rank-specific subdirectory for sharded checkpoints
        self.rank = dist_utils.get_global_rank()
        self.world_size = dist_utils.get_world_size()

        # Track saved checkpoints for rotation
        self.saved_checkpoints: typing.List[pathlib.Path] = []

    def save_checkpoint(
        self, ckpt_data: checkpoint_data.CheckpointData, config: core.TrainerConfig
    ) -> None:
        """Save a sharded FSDP checkpoint.

        Each rank saves its own shard of the model and optimizer states.
        Note: The ckpt_data model and optimizer state dicts are ignored -
        we extract them directly within the FSDP context to ensure proper sharding.

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

        logger.info(f"[Rank {self.rank}] Saving FSDP checkpoint to {checkpoint_dir}")

        # Save config first (only rank 0)
        if self.rank == 0:
            config_path = checkpoint_dir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config.model_dump(), f, default_flow_style=False)

        # Get FSDP state dict with proper configuration
        # The optimizer state is automatically sharded when using FSDP
        with fsdp.state_dict_type(
            self.model,
            fsdp.StateDictType.SHARDED_STATE_DICT,
            fsdp.ShardedStateDictConfig(offload_to_cpu=True),
        ):
            # Get sharded model state dict
            sharded_model_state_dict = self.model.state_dict()

            # Get sharded optimizer state dict
            # When using FSDP, the optimizer state is automatically sharded
            # to match the model sharding
            sharded_optimizer_state_dict = self.optimizer.state_dict()

        # Save each component as a sharded file
        # Each rank saves its shard with rank suffix
        model_path = checkpoint_dir / f"model_rank{self.rank:04d}.pt"
        torch.save(sharded_model_state_dict, model_path)

        optim_path = checkpoint_dir / f"optim_rank{self.rank:04d}.pt"
        torch.save(sharded_optimizer_state_dict, optim_path)

        # Non-sharded components (same across ranks, but each rank saves for simplicity)
        scheduler_path = checkpoint_dir / "scheduler.pt"
        if self.rank == 0:  # Only rank 0 saves non-sharded components
            torch.save(ckpt_data.scheduler_state, scheduler_path)

            train_path = checkpoint_dir / "train.pt"
            torch.save(ckpt_data.training_state, train_path)

        # Synchronize across all ranks
        dist_utils.barrier()

        # Only rank 0 manages symlinks and rotation
        if self.rank == 0:
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
                logger.info(f"New best model! Linking to {checkpoint_name}")
                best_link.symlink_to(checkpoint_name)

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to keep only the last N."""
        if self.config.keep_last_n <= 0:
            return  # Keep all checkpoints

        while len(self.saved_checkpoints) > self.config.keep_last_n:
            old_checkpoint_dir = self.saved_checkpoints.pop(0)
            if old_checkpoint_dir.exists():
                logger.info(f"Removing old checkpoint directory: {old_checkpoint_dir}")
                shutil.rmtree(old_checkpoint_dir)

    def find_resume_checkpoint(self) -> typing.Optional[pathlib.Path]:
        """Find checkpoint to resume from based on config.

        Returns:
            Path to checkpoint directory, or None if not found
        """
        if self.config.resume_from:
            # Explicit path provided (should be a directory for FSDP)
            resume_path = pathlib.Path(self.config.resume_from)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            if not resume_path.is_dir():
                raise ValueError(f"FSDP checkpoint should be a directory: {resume_path}")
            return resume_path

        # Try to find latest checkpoint
        latest_link = self.save_dir / "latest"
        if latest_link.exists():
            return latest_link.resolve()

        # No checkpoint to resume from
        return None

    def load_checkpoint(self, device: str) -> typing.Optional[checkpoint_data.CheckpointData]:
        """Load a sharded FSDP checkpoint for resuming.

        Each rank loads its own shard of the model and optimizer states.

        Args:
            device: Device to load tensors to

        Returns:
            CheckpointData object, or None if no checkpoint found
        """
        checkpoint_dir = self.find_resume_checkpoint()
        if checkpoint_dir is None:
            logger.info("No checkpoint found to resume from")
            return None

        logger.info(f"[Rank {self.rank}] Loading FSDP checkpoint from {checkpoint_dir}")

        # Load sharded components
        model_path = checkpoint_dir / f"model_rank{self.rank:04d}.pt"
        optim_path = checkpoint_dir / f"optim_rank{self.rank:04d}.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"[Rank {self.rank}] Model checkpoint shard not found: {model_path}"
            )
        if not optim_path.exists():
            raise FileNotFoundError(
                f"[Rank {self.rank}] Optimizer checkpoint shard not found: {optim_path}"
            )

        # Load sharded state dicts
        # Use weights_only=False for backward compatibility with PyTorch 2.6+
        sharded_model_state = torch.load(model_path, map_location=device, weights_only=False)
        sharded_optimizer_state = torch.load(optim_path, map_location=device, weights_only=False)

        # Load non-sharded components (from rank 0's save)
        scheduler_path = checkpoint_dir / "scheduler.pt"
        train_path = checkpoint_dir / "train.pt"

        scheduler_state = (
            torch.load(scheduler_path, map_location=device, weights_only=False)
            if scheduler_path.exists()
            else {}
        )
        training_state = (
            torch.load(train_path, map_location=device, weights_only=False)
            if train_path.exists()
            else {}
        )

        # Create CheckpointData with raw dicts
        ckpt_data = checkpoint_data.CheckpointData(
            model_state=sharded_model_state,
            optimizer_state=sharded_optimizer_state,
            scheduler_state=scheduler_state,
            training_state=training_state,
        )

        # Load state dict into FSDP model and optimizer with proper configuration
        with fsdp.state_dict_type(
            self.model,
            fsdp.StateDictType.SHARDED_STATE_DICT,
            fsdp.ShardedStateDictConfig(offload_to_cpu=True),
        ):
            # Load model state
            self.model.load_state_dict(ckpt_data.model_state)

            # Load optimizer state
            if ckpt_data.optimizer_state:
                self.optimizer.load_state_dict(ckpt_data.optimizer_state)

        # Update best val loss if resuming
        if "best_val_loss" in ckpt_data.training_state:
            self.best_val_loss = ckpt_data.training_state["best_val_loss"]

        return ckpt_data

    def save_model_only(
        self,
        model_state_dict: typing.Dict[str, torch.Tensor],
        metadata: typing.Dict[str, typing.Any],
    ) -> None:
        """Save just the model weights (no optimizer state) in consolidated format.

        This gathers the full model from all ranks and saves it as a single file
        on rank 0. Useful for deployment/inference where you don't need sharded format.

        Note: The model_state_dict parameter is ignored for FSDP. We extract the
        full state dict directly from the FSDP-wrapped model using FSDP's
        gathering functionality.

        Args:
            model_state_dict: Model state dictionary (ignored for FSDP)
            metadata: Additional metadata (config, iteration, loss, etc.)
        """
        iter_num = metadata.get("iteration", 0)
        val_loss = metadata.get("val_loss", 0.0)

        # Use FULL_STATE_DICT to gather the complete model on rank 0
        with fsdp.state_dict_type(
            self.model,
            fsdp.StateDictType.FULL_STATE_DICT,
            fsdp.FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            full_state_dict = self.model.state_dict()

            # Only rank 0 saves the consolidated model
            if self.rank == 0:
                final_path = self.save_dir / f"model_iter{iter_num}_loss{val_loss:.4f}.pt"

                model_state = {"model": full_state_dict, **metadata}

                logger.info(f"Saving consolidated model to {final_path}")
                torch.save(model_state, final_path)
