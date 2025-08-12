# Third Party
import pydantic
import yaml

# Project
from pretraining.configs.model import transformer
from pretraining.configs.model.architectures import base
from pretraining.configs.model.components import attention
from pretraining.configs.model.components import normalization
from pretraining.configs.model.components import position


class Llama3Config(base.BaseLLMConfig):
    """Configuration for Llama architecture (3.1 and similar)."""

    transformer: transformer.Llama3TransformerConfig

    @pydantic.model_validator(mode="after")
    def validate_config(self) -> "Llama3Config":
        """Validate Llama specific configuration."""
        # Llama requires RoPE config
        if self.transformer.rope is None:
            raise ValueError("Llama architecture requires RoPE config")

        # Llama doesn't traditionally use tied embeddings
        if self.output_head.tie_word_embeddings:
            raise ValueError(
                "Llama traditionally uses separate output projection (tie_word_embeddings=False)"
            )

        # Llama doesn't traditionally use bias in transformer layers
        if self.transformer.bias:
            raise ValueError("Llama traditionally uses no bias in transformer layers (bias=False)")

        # Ensure we're using RMSNorm (Llama's traditional normalization)
        if not isinstance(self.transformer.normalization, normalization.RMSNormConfig):
            raise ValueError("Llama traditionally uses RMSNorm normalization")

        # Ensure we're using GroupedQueryAttention (for GQA support)
        if not isinstance(self.transformer.attention, attention.GroupedQueryAttentionConfig):
            raise ValueError("Llama traditionally uses GroupedQueryAttention for GQA support")

        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Llama3Config":
        """Load Llama3Config from a YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        model_dict = config_dict["model"]

        # Transform transformer components to concrete types
        trans_dict = model_dict["transformer"]
        trans_dict["normalization"] = normalization.RMSNormConfig.model_validate(
            trans_dict["normalization"]
        )
        trans_dict["attention"] = attention.GroupedQueryAttentionConfig.model_validate(
            trans_dict["attention"]
        )

        # Parse RoPE config with optional scaling
        rope_dict = trans_dict["rope"]
        if "scaling" in rope_dict and rope_dict["scaling"] is not None:
            rope_dict["scaling"] = position.RoPEScalingConfig.model_validate(rope_dict["scaling"])
        trans_dict["rope"] = position.RoPEConfig.model_validate(rope_dict)

        # Create Llama3TransformerConfig
        model_dict["transformer"] = transformer.Llama3TransformerConfig.model_validate(trans_dict)

        return cls.model_validate(model_dict)
