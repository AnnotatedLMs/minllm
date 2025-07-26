# Third Party
import pydantic
import yaml

# Project
from pretraining.configs.model.architectures import base
from pretraining.configs.model.components import attention
from pretraining.configs.model.components import normalization
from pretraining.configs.model.components import position


class Llama3Config(base.BaseLLMConfig):
    """Configuration for Llama architecture (3.1 and similar)."""

    # No additional fields - Llama uses all base fields

    @pydantic.model_validator(mode="after")
    def validate_config(self) -> "Llama3Config":
        """Validate Llama specific configuration."""
        # Llama requires RoPE config
        if self.transformer.rope is None:
            raise ValueError("Llama architecture requires RoPE config")

        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Llama3Config":
        """Load Llama3Config from a YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        model_dict = config_dict["model"]

        # Transform transformer components to concrete types
        trans_dict = model_dict["transformer"]
        trans_dict["normalization"] = normalization.RMSNormConfig(**trans_dict["normalization"])
        trans_dict["attention"] = attention.GroupedQueryAttentionConfig(**trans_dict["attention"])

        # Parse RoPE config with optional scaling
        rope_dict = trans_dict["rope"]
        if "scaling" in rope_dict and rope_dict["scaling"] is not None:
            rope_dict["scaling"] = position.RoPEScalingConfig(**rope_dict["scaling"])
        trans_dict["rope"] = position.RoPEConfig(**rope_dict)

        return cls(**model_dict)
