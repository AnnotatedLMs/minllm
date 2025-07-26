# Standard Library
import logging
import pathlib
import typing

# Third Party
import jaxtyping
import torch

# Project
from pretraining.configs.training import batch_configs
from pretraining.configs.training import data_configs
from pretraining.configs.training import trainer_configs
from pretraining.data import dataset
from pretraining.utils.training import distributed

logger = logging.getLogger(__name__)


class PretrainDataLoader:
    """
    DataLoader for GPT pretraining.

    Handles:
    - Memory-mapped data loading
    - Efficient GPU transfer with pinned memory
    - Train/val split management
    """

    def __init__(
        self,
        data_config: data_configs.DataConfig,
        batch_config: batch_configs.BatchConfig,
        device: torch.device,
    ):
        self.data_config = data_config
        self.batch_config = batch_config
        self.device = device

        # Store commonly accessed values
        self.batch_size = batch_config.batch_size
        self.block_size = batch_config.sequence_length
        self.device_type = self.device.type

        # Get distributed info
        self.rank = distributed.get_global_rank()
        self.world_size = distributed.get_world_size()

        # Initialize datasets
        data_dir = pathlib.Path(data_config.data_dir)
        self.train_dataset = dataset.PretrainDataset(data_dir, "train")
        self.val_dataset = dataset.PretrainDataset(data_dir, "val")

        # For distributed training, calculate the effective dataset size per rank
        self.train_tokens_per_rank = len(self.train_dataset) // self.world_size
        self.val_tokens_per_rank = len(self.val_dataset) // self.world_size

        # Track iteration state for checkpoint resume and data position
        self.train_iteration = 0
        self.val_iteration = 0

        logger.info(f"Train dataset: {len(self.train_dataset):,} tokens")
        logger.info(f"Val dataset: {len(self.val_dataset):,} tokens")
        if self.world_size > 1:
            logger.info(
                f"Tokens per rank: train={self.train_tokens_per_rank:,}, val={self.val_tokens_per_rank:,}"
            )

    def _select_dataset(self, split: typing.Literal["train", "val"]) -> dataset.PretrainDataset:
        """Select the appropriate dataset based on split."""
        return self.train_dataset if split == "train" else self.val_dataset

    def _transfer_batch_to_device(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Transfer batch tensors to the target device with optimizations."""
        if self.device_type == "cuda" and self.data_config.pin_memory:
            # Pin memory for async GPU transfer
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def get_num_tokens(self, split: typing.Literal["train", "val"]) -> int:
        """Get the total number of tokens in a split."""
        dataset = self.train_dataset if split == "train" else self.val_dataset
        return len(dataset)

    def get_num_batches(self, split: typing.Literal["train", "val"]) -> int:
        """Estimate number of non-overlapping batches possible."""
        num_tokens = self.get_num_tokens(split)
        tokens_per_batch = self.batch_size * self.block_size
        return max(1, num_tokens // tokens_per_batch)

    def get_batch(
        self, split: typing.Literal["train", "val"]
    ) -> typing.Tuple[
        jaxtyping.Int[torch.Tensor, "batch seq"], jaxtyping.Int[torch.Tensor, "batch seq"]
    ]:
        """
        Get a batch of data for training or validation.

        Args:
            split: 'train' or 'val'

        Returns:
            x: Input token ids of shape (batch_size, block_size)
            y: Target token ids of shape (batch_size, block_size)
        """

        dataset = self._select_dataset(split)
        x, y = dataset.sample_batch(self.batch_size, self.block_size)
        x, y = self._transfer_batch_to_device(x, y)

        # Increment iteration counter to ensure different data for gradient accumulation
        if split == "train":
            self.train_iteration += 1
        else:
            self.val_iteration += 1

        return x, y

    def get_state_dict(self) -> typing.Dict[str, typing.Any]:
        """Get state dict for checkpointing."""
        return {
            "train_iteration": self.train_iteration,
            "val_iteration": self.val_iteration,
        }

    def load_state_dict(self, state_dict: typing.Dict[str, typing.Any]) -> None:
        """Load state dict for checkpoint resume."""
        self.train_iteration = state_dict.get("train_iteration", 0)
        self.val_iteration = state_dict.get("val_iteration", 0)
        logger.info(
            f"Resumed dataloader at train_iter={self.train_iteration}, val_iter={self.val_iteration}"
        )

    @classmethod
    def from_training_config(
        cls,
        training_config: "trainer_configs.TrainingLoopConfig",
        device: torch.device,
    ) -> "PretrainDataLoader":
        """Create dataloader from full training configuration."""
        return cls(
            data_config=training_config.data,
            batch_config=training_config.batch,
            device=device,
        )
