# Standard Library
import dataclasses
import typing

# Project
# Local
from pretraining.configs.llm import llm_configs
from pretraining.configs.training import trainer_configs


@dataclasses.dataclass
class TrainerConfig:
    """
    Root configuration that combines model and training configurations.

    This is the top-level configuration object that contains all settings
    needed for a complete training run. Fields are ordered by their
    hierarchical relationship:
    1. Model configuration (what to train)
    2. Training configuration (how to train it)
    """

    # 1. Model configuration (defines the LLM architecture)
    llm: typing.Union[llm_configs.GPT2Config, llm_configs.LlamaConfig, llm_configs.DeepSeekConfig]

    # 2. Training configuration (defines the training process)
    training: trainer_configs.TrainingLoopConfig

    def __post_init__(self):
        """Validate consistency between model and training configurations."""
        # Validate batch sequence length matches model block size
        if self.training.batch.sequence_length != self.llm.transformer_config.block_size:
            raise ValueError(
                f"Batch sequence_length ({self.training.batch.sequence_length}) must match "
                f"model block_size ({self.llm.transformer_config.block_size})"
            )

        # Validate MoE training config presence
        if (
            self.llm.transformer_config.moe_config is not None
            and self.training.moe_training is None
        ):
            raise ValueError("Model has MoE config but moe_training config is missing")
        if (
            self.llm.transformer_config.moe_config is None
            and self.training.moe_training is not None
        ):
            raise ValueError("Model has no MoE config but moe_training config is provided")

        # Validate MTP training config presence (only DeepSeek has MTP)
        if hasattr(self.llm, "mtp_config"):
            if self.llm.mtp_config is not None and self.training.mtp_training is None:
                raise ValueError("Model has MTP config but mtp_training config is missing")
            if self.llm.mtp_config is None and self.training.mtp_training is not None:
                raise ValueError("Model has no MTP config but mtp_training config is provided")
        elif self.training.mtp_training is not None:
            raise ValueError("Model does not support MTP but mtp_training config is provided")
