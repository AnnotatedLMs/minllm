# Standard Library
import typing

# Third Party
import jaxtyping
import torch

# Project
from pretraining.configs.model.architectures import base as model_base


class DataCollator:
    """Base data collator for language modeling.

    Handles standard next-token prediction targets.
    Works with PyTorch DataLoader to process batches of samples.
    """

    def __call__(
        self,
        batch: typing.List[typing.Dict[str, typing.Any]],
    ) -> typing.Dict[str, torch.Tensor]:
        """Process batch of samples from DataLoader.

        Args:
            batch: List of sample dictionaries from dataset

        Returns:
            Dictionary with model inputs
        """
        # Extract input_ids from each sample
        input_ids = self._extract_input_ids(batch)

        # Stack into batch tensor
        input_ids = self._stack_tensors(input_ids)

        # Create next-token prediction targets
        labels = self._create_labels(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def _extract_input_ids(
        self, batch: typing.List[typing.Dict[str, typing.Any]]
    ) -> typing.List[torch.Tensor]:
        """Extract input_ids from each sample."""
        return [sample["input_ids"] for sample in batch]

    def _stack_tensors(self, tensors: typing.List[torch.Tensor]) -> torch.Tensor:
        """Stack list of tensors into batch tensor."""
        return torch.stack(tensors, dim=0)

    def _create_labels(
        self, input_ids: jaxtyping.Int[torch.Tensor, "batch seq"]
    ) -> jaxtyping.Int[torch.Tensor, "batch seq"]:
        """Create next-token prediction labels by shifting input_ids."""
        # For standard LM, labels are just shifted input_ids
        # Model will internally shift and compute loss
        return input_ids.clone()


class MTPDataCollator(DataCollator):
    """Data collator for multi-token prediction (MTP).

    In addition to standard next-token prediction, MTP predicts multiple
    future tokens at each position. For example, with mtp_depth=3:

    Given input: [A, B, C, D, E]
    - Standard prediction at pos 0: predict B (1 token ahead)
    - MTP depth 0 at pos 0: predict B (1 token ahead)
    - MTP depth 1 at pos 0: predict C (2 tokens ahead)
    - MTP depth 2 at pos 0: predict D (3 tokens ahead)

    This helps the model learn longer-range dependencies.
    """

    def __init__(self, mtp_depth: int):
        """Initialize MTP collator.

        Args:
            mtp_depth: Number of future tokens to predict at each position
        """
        self.mtp_depth = mtp_depth

    def __call__(
        self,
        batch: typing.List[typing.Dict[str, typing.Any]],
    ) -> typing.Dict[str, torch.Tensor]:
        """Process batch with both standard and MTP targets.

        Args:
            batch: List of sample dictionaries from dataset

        Returns:
            Dictionary containing:
            - input_ids: Input tokens [batch, seq]
            - labels: Standard next-token targets [batch, seq]
            - mtp_targets: Multi-token prediction targets [batch, depth, seq]
        """
        # Extract and stack input_ids
        input_ids = self._extract_input_ids(batch)
        input_ids = self._stack_tensors(input_ids)

        # Create standard next-token labels (same as base collator)
        labels = self._create_labels(input_ids)

        # Create MTP targets for predicting multiple future tokens
        mtp_targets = self._create_mtp_targets(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,  # Standard next-token targets
            "mtp_targets": mtp_targets,  # Multi-token prediction targets
        }

    def _create_mtp_targets(
        self,
        input_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
    ) -> jaxtyping.Int[torch.Tensor, "batch depth seq"]:
        """Create targets for predicting multiple future tokens.

        For mtp_depth=3, this creates targets for predicting 1, 2, and 3 tokens ahead.
        """
        batch_size, seq_len = input_ids.shape

        # Initialize with ignore index
        mtp_targets = self._initialize_mtp_targets(
            batch_size, seq_len, input_ids.dtype, input_ids.device
        )

        # Fill targets for each depth
        for depth in range(self.mtp_depth):
            self._fill_targets_for_depth(mtp_targets, input_ids, depth)

        return mtp_targets

    def _initialize_mtp_targets(
        self,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Initialize MTP targets tensor with ignore index."""
        return torch.full(
            (batch_size, self.mtp_depth, seq_len),
            -100,  # Standard ignore index for CrossEntropyLoss
            dtype=dtype,
            device=device,
        )

    def _fill_targets_for_depth(
        self,
        mtp_targets: torch.Tensor,
        input_ids: torch.Tensor,
        depth: int,
    ) -> None:
        """Fill targets for a specific prediction depth.

        For depth k, we predict tokens that are k+1 positions ahead.
        """
        seq_len = input_ids.shape[1]
        tokens_ahead = depth + 1

        if tokens_ahead < seq_len:
            # At position i, predict token at position i + tokens_ahead
            # We can only predict up to position (seq_len - tokens_ahead)
            valid_positions = seq_len - tokens_ahead
            mtp_targets[:, depth, :valid_positions] = input_ids[:, tokens_ahead:]


def build_collator(
    model_config: model_base.BaseLLMConfig,
) -> DataCollator:
    """Build appropriate collator based on model configuration.

    Args:
        model_config: Model configuration

    Returns:
        Data collator instance
    """
    # Check if model has MTP configuration
    if hasattr(model_config, "mtp") and model_config.mtp is not None:
        return MTPDataCollator(mtp_depth=model_config.mtp.depth)

    return DataCollator()
