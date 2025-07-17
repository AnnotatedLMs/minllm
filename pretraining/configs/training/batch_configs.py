# Standard Library
import dataclasses


@dataclasses.dataclass
class BatchConfig:
    """
    Configuration for batch processing during training.

    Consumed by: DataLoader, gradient accumulation logic, and training loop.
    """

    batch_size: int  # Number of sequences per batch
    sequence_length: int  # Length of each sequence (must match model.block_size)
    gradient_accumulation_steps: int  # Number of batches to accumulate before updating weights
