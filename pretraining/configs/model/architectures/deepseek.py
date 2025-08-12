# Third Party
import pydantic
import yaml

# Project
from pretraining.configs.model import transformer
from pretraining.configs.model.architectures import base
from pretraining.configs.model.components import attention
from pretraining.configs.model.components import heads
from pretraining.configs.model.components import normalization
from pretraining.configs.model.components import position


class DeepSeek3Config(base.BaseLLMConfig):
    """Configuration for DeepSeek-V3 architecture."""

    # Override with DeepSeek 3 specific transformer config
    transformer: transformer.DeepSeek3TransformerConfig

    # Multi-token prediction heads (predicts multiple future tokens)
    mtp: heads.MultiTokenPredictionConfig

    @pydantic.model_validator(mode="after")
    def validate_config(self) -> "DeepSeek3Config":
        """Validate DeepSeek specific configuration."""
        # DeepSeek requires RoPE config
        if self.transformer.rope is None:
            raise ValueError("DeepSeek architecture requires RoPE config")

        # DeepSeek requires MoE config (no standard FFN)
        if self.transformer.moe is None:
            raise ValueError("DeepSeek architecture requires MoE config")
        if self.transformer.ffn is not None:
            raise ValueError("DeepSeek uses MoE instead of standard FFN")

        # DeepSeek doesn't traditionally use tied embeddings
        if self.output_head.tie_word_embeddings:
            raise ValueError(
                "DeepSeek traditionally uses separate output projection (tie_word_embeddings=False)"
            )

        # DeepSeek doesn't traditionally use bias in transformer layers
        if self.transformer.bias:
            raise ValueError(
                "DeepSeek traditionally uses no bias in transformer layers (bias=False)"
            )

        # Ensure we're using RMSNorm
        if not isinstance(self.transformer.normalization, normalization.RMSNormConfig):
            raise ValueError("DeepSeek traditionally uses RMSNorm normalization")

        # Ensure we're using MultiHeadLatentAttention (MLA)
        if not isinstance(self.transformer.attention, attention.MultiHeadLatentAttentionConfig):
            raise ValueError("DeepSeek requires MultiHeadLatentAttention (MLA) config")

        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DeepSeek3Config":
        """Load DeepSeek3Config from a YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        model_dict = config_dict["model"]

        # Transform transformer components to concrete types
        trans_dict = model_dict["transformer"]
        trans_dict["normalization"] = normalization.RMSNormConfig.model_validate(
            trans_dict["normalization"]
        )
        trans_dict["attention"] = attention.MultiHeadLatentAttentionConfig.model_validate(
            trans_dict["attention"]
        )

        # Parse RoPE config
        trans_dict["rope"] = position.RoPEConfig.model_validate(trans_dict["rope"])

        # Parse MTP config
        model_dict["mtp"] = heads.MultiTokenPredictionConfig.model_validate(model_dict["mtp"])

        # Create DeepSeek3TransformerConfig
        model_dict["transformer"] = transformer.DeepSeek3TransformerConfig.model_validate(
            trans_dict
        )

        return cls.model_validate(model_dict)
