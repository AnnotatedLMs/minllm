# Standard Library
import typing

# Third Party
import torch
from torch import nn

# Project
from pretraining.configs.training import execution_configs
from pretraining.utils.training import activation_checkpointing


class BlockGroup(nn.ModuleList):
    """
    Groups multiple transformer blocks for FSDP wrapping and activation checkpointing.

    This container allows treating multiple blocks as a single unit for FSDP while
    still applying activation checkpointing at the individual block level based on
    the configured strategy.
    """

    def __init__(
        self,
        blocks: typing.List[nn.Module],
        layer_offset: int = 0,
        activation_checkpointing_strategy: typing.Optional[
            execution_configs.ActivationCheckpointingStrategy
        ] = None,
        activation_checkpointing_reentrant: bool = False,
    ):
        """
        Initialize a BlockGroup.

        Args:
            blocks: List of transformer blocks to group
            layer_offset: Offset for layer indices (for activation checkpointing decisions)
            activation_checkpointing_strategy: Strategy for activation checkpointing
            activation_checkpointing_reentrant: Whether to use reentrant checkpointing
        """
        super().__init__(blocks)
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy = activation_checkpointing_strategy
        self.activation_checkpointing_reentrant = activation_checkpointing_reentrant
        self._activation_checkpoint_fn: typing.Optional[typing.Callable] = None

        if (
            activation_checkpointing_strategy
            and activation_checkpointing_strategy
            != execution_configs.ActivationCheckpointingStrategy.NONE
        ):
            self._setup_activation_checkpointing()

    def _setup_activation_checkpointing(self) -> None:
        """Setup activation checkpointing function."""
        self._activation_checkpoint_fn = activation_checkpointing.get_activation_checkpoint_fn(
            reentrant=self.activation_checkpointing_reentrant,
            preserve_rng_state=True,  # Conservative default for dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: typing.Any,
    ) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, typing.Any]]:
        """
        Forward pass through all blocks in the group.

        Args:
            x: Input tensor
            **kwargs: Additional arguments to pass to blocks (attention_mask, cache, etc.)

        Returns:
            Output from the last block (could be tensor or tuple with cache)
        """
        # Track any caches returned by blocks
        output = x

        for block_idx, block in enumerate(self):
            global_block_idx = self.layer_offset + block_idx

            # Extract hidden states if previous block returned a tuple
            block_input = output[0] if isinstance(output, tuple) else output

            # Determine if this block should be checkpointed
            if activation_checkpointing.should_checkpoint_block(
                self.activation_checkpointing_strategy, global_block_idx
            ):
                if self._activation_checkpoint_fn is not None:
                    # Use activation checkpointing for this block
                    # Note: checkpoint function handles both single return and tuple returns
                    output = self._activation_checkpoint_fn(block, block_input, **kwargs)
                else:
                    # Fallback if checkpoint function not set up
                    output = block(block_input, **kwargs)
            else:
                # Normal forward pass without checkpointing
                output = block(block_input, **kwargs)

        return output

    def set_activation_checkpointing(
        self,
        strategy: typing.Optional[execution_configs.ActivationCheckpointingStrategy],
        reentrant: bool = False,
    ) -> None:
        """
        Update activation checkpointing configuration.

        Args:
            strategy: New activation checkpointing strategy
            reentrant: Whether to use reentrant checkpointing
        """
        self.activation_checkpointing_strategy = strategy
        self.activation_checkpointing_reentrant = reentrant

        if strategy and strategy != execution_configs.ActivationCheckpointingStrategy.NONE:
            self._setup_activation_checkpointing()
        else:
            self._activation_checkpoint_fn = None
