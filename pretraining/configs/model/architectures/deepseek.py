# Third Party
import pydantic
import yaml

# Project
from pretraining.configs.model.architectures import base
from pretraining.configs.model.components import attention
from pretraining.configs.model.components import heads
from pretraining.configs.model.components import normalization
from pretraining.configs.model.components import position


class DeepSeek3Config(base.BaseLLMConfig):
    """Configuration for DeepSeek-V3 architecture."""

    # Multi-token prediction heads (predicts multiple future tokens)
    mtp: heads.MultiTokenPredictionConfig

    @pydantic.model_validator(mode="after")
    def validate_config(self) -> "DeepSeek3Config":
        """Validate DeepSeek specific configuration."""
        # DeepSeek requires RoPE config
        if self.transformer.rope is None:
            raise ValueError("DeepSeek architecture requires RoPE config")

        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DeepSeek3Config":
        """Load DeepSeek3Config from a YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        model_dict = config_dict["model"]

        # Transform transformer components to concrete types
        trans_dict = model_dict["transformer"]
        trans_dict["normalization"] = normalization.RMSNormConfig(**trans_dict["normalization"])
        trans_dict["attention"] = attention.MultiHeadLatentAttentionConfig(
            **trans_dict["attention"]
        )

        # Parse RoPE config
        trans_dict["rope"] = position.RoPEConfig(**trans_dict["rope"])

        return cls(**model_dict)
