# Project
from pretraining.configs import base


class BaseNormalizationConfig(base.BaseConfig):
    """Base configuration for normalization layers."""

    norm_eps: float


class LayerNormConfig(BaseNormalizationConfig):
    """Configuration for LayerNorm."""

    bias: bool  # Whether to use bias in LayerNorm


class RMSNormConfig(BaseNormalizationConfig):
    """Configuration for RMSNorm (used in Llama models)."""

    pass
