# Third Party
import pydantic
import yaml

# Project
from pretraining.configs.model import initialization
from pretraining.configs.model import transformer
from pretraining.configs.model.architectures import base
from pretraining.configs.model.components import attention
from pretraining.configs.model.components import embeddings
from pretraining.configs.model.components import normalization


class GPT2Config(base.BaseLLMConfig):
    """Configuration for GPT-2 architecture."""

    transformer: transformer.GPT2TransformerConfig

    position_embedding: embeddings.LearnedPositionEmbeddingConfig

    weight_init: initialization.GPT2InitConfig

    @pydantic.model_validator(mode="after")
    def validate_config(self) -> "GPT2Config":
        """Validate GPT-2 specific configuration."""
        # Ensure position embedding dim matches transformer hidden dim
        if self.position_embedding.embedding_dim != self.transformer.hidden_dim:
            raise ValueError(
                f"position embedding_dim ({self.position_embedding.embedding_dim}) must match "
                f"transformer hidden_dim ({self.transformer.hidden_dim})"
            )

        if self.transformer.rope is not None:
            raise ValueError("GPT-2 architecture should not have RoPE config")

        if not self.output_head.tie_word_embeddings:
            raise ValueError("GPT-2 requires tied embeddings (tie_word_embeddings=True)")

        if self.output_head.lm_head_bias:
            raise ValueError("GPT-2 cannot use lm_head_bias with tied embeddings")

        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GPT2Config":
        """Load GPT2Config from a YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        model_dict = config_dict["model"]

        model_dict["weight_init"] = initialization.GPT2InitConfig.model_validate(
            model_dict["weight_init"]
        )

        trans_dict = model_dict["transformer"]
        trans_dict["normalization"] = normalization.LayerNormConfig.model_validate(
            trans_dict["normalization"]
        )
        trans_dict["attention"] = attention.MultiHeadAttentionConfig.model_validate(
            trans_dict["attention"]
        )

        model_dict["transformer"] = transformer.GPT2TransformerConfig.model_validate(trans_dict)

        return cls.model_validate(model_dict)
