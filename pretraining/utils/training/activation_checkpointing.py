# Standard Library
import functools
import typing

# Third Party
from torch.utils import checkpoint

# Project
from pretraining.configs.training import execution_configs


def should_checkpoint_block(
    strategy: typing.Optional[execution_configs.ActivationCheckpointingStrategy],
    block_idx: int,
) -> bool:
    """
    Determine if a specific block should use activation checkpointing.

    Args:
        strategy: The activation checkpointing strategy
        block_idx: The index of the block (0-based)

    Returns:
        Whether this block should be checkpointed
    """
    if strategy is None or strategy == execution_configs.ActivationCheckpointingStrategy.NONE:
        return False
    elif strategy == execution_configs.ActivationCheckpointingStrategy.WHOLE_LAYER:
        return True
    elif strategy == execution_configs.ActivationCheckpointingStrategy.ONE_IN_TWO:
        return block_idx % 2 == 0
    elif strategy == execution_configs.ActivationCheckpointingStrategy.ONE_IN_THREE:
        return block_idx % 3 == 0
    elif strategy == execution_configs.ActivationCheckpointingStrategy.ONE_IN_FOUR:
        return block_idx % 4 == 0
    elif strategy == execution_configs.ActivationCheckpointingStrategy.ONE_IN_EIGHT:
        return block_idx % 8 == 0
    elif strategy == execution_configs.ActivationCheckpointingStrategy.TWO_IN_THREE:
        return block_idx % 3 != 0
    elif strategy == execution_configs.ActivationCheckpointingStrategy.THREE_IN_FOUR:
        return block_idx % 4 != 0
    elif strategy == execution_configs.ActivationCheckpointingStrategy.FINE_GRAINED:
        # Fine-grained strategy: checkpoint expensive operations like attention
        # but not cheap ones like layer norm. For now, default to every other block.
        # This should be customized per architecture.
        return block_idx % 2 == 0
    else:
        return False


def get_activation_checkpoint_fn(
    reentrant: bool = False,
    preserve_rng_state: bool = True,
) -> typing.Callable:
    """
    Get a configured activation checkpointing function.

    Args:
        reentrant: Whether to use reentrant checkpointing (slower but more compatible)
        preserve_rng_state: Whether to preserve RNG state (needed when using dropout)

    Returns:
        Configured checkpoint function
    """
    return functools.partial(
        checkpoint.checkpoint,
        use_reentrant=reentrant,
        preserve_rng_state=preserve_rng_state,
    )
