# Third Party
import pydantic

# Project
from pretraining.configs import base
from pretraining.configs.model.architectures import base as base_llm
from pretraining.configs.training import trainer_configs


class TrainerConfig(base.BaseConfig):
    """
    Root configuration that combines model and training configurations.

    This is the top-level configuration object that contains all settings
    needed for a complete training run. Fields are ordered by their
    hierarchical relationship:
    1. Model configuration (what to train)
    2. Training configuration (how to train it)
    """

    # 1. Model configuration (defines the LLM architecture)
    llm: base_llm.BaseLLMConfig

    # 2. Training configuration (defines the training process)
    training: trainer_configs.TrainingLoopConfig

    @pydantic.model_validator(mode="after")
    def validate_config_consistency(self):
        """Validate consistency between model and training configurations."""
        # Validate batch sequence length matches model block size
        if self.training.batch.sequence_length != self.llm.transformer.block_size:
            raise ValueError(
                f"Batch sequence_length ({self.training.batch.sequence_length}) must match "
                f"model block_size ({self.llm.transformer.block_size})"
            )

        # Validate MoE training config presence
        if self.llm.transformer.moe is not None and self.training.moe_training is None:
            raise ValueError("Model has MoE config but moe_training config is missing")
        if self.llm.transformer.moe is None and self.training.moe_training is not None:
            raise ValueError("Model has no MoE config but moe_training config is provided")

        # Validate MTP training config presence (only DeepSeek has MTP)
        if hasattr(self.llm, "mtp"):
            if self.llm.mtp is not None and self.training.mtp_training is None:
                raise ValueError("Model has MTP config but mtp_training config is missing")
            if self.llm.mtp is None and self.training.mtp_training is not None:
                raise ValueError("Model has no MTP config but mtp_training config is provided")
        elif self.training.mtp_training is not None:
            raise ValueError("Model does not support MTP but mtp_training config is provided")

        return self
