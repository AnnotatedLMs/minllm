# Third Party
import pydantic

# Project
from pretraining.configs import base


class BatchConfig(base.BaseConfig):
    """
    Configuration for batch processing during training.

    Consumed by: DataLoader, gradient accumulation logic, and training loop.
    """

    batch_size: int = pydantic.Field(gt=0, description="Number of sequences per batch")
    sequence_length: int = pydantic.Field(
        gt=0, description="Length of each sequence (must match model.block_size)"
    )
    gradient_accumulation_steps: int = pydantic.Field(
        gt=0, description="Number of batches to accumulate before updating weights"
    )
