# Standard Library
import typing

# Third Party
import pydantic

# Project
from pretraining.configs import base
from pretraining.configs.model import initialization
from pretraining.configs.model import transformer
from pretraining.configs.model.components import embeddings
from pretraining.configs.model.components import heads


class BaseLLMConfig(base.BaseConfig):
    """Base configuration shared by many LLM architectures."""

    token_embedding: embeddings.TokenEmbeddingConfig
    transformer: transformer.TransformerConfig
    output_head: heads.OutputHeadConfig
    weight_init: typing.Optional[initialization.BaseInitializationConfig] = None

    @pydantic.model_validator(mode="after")
    def validate_base_config(self):
        """Validate common configuration consistency."""
        # Ensure embedding dim matches transformer hidden dim
        if self.token_embedding.embedding_dim != self.transformer.hidden_dim:
            raise ValueError(
                f"token embedding_dim ({self.token_embedding.embedding_dim}) must match "
                f"transformer hidden_dim ({self.transformer.hidden_dim})"
            )

        # Ensure vocab sizes match
        if self.token_embedding.vocab_size != self.transformer.vocab_size:
            raise ValueError(
                f"token embedding vocab_size ({self.token_embedding.vocab_size}) must match "
                f"transformer vocab_size ({self.transformer.vocab_size})"
            )

        return self
