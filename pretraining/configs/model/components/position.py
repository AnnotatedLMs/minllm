# Standard Library
import typing

# Project
from pretraining.configs import base


class RoPEScalingConfig(base.BaseConfig):
    """Configuration for RoPE scaling (context length extension)."""

    scale_factor: float  # Factor to scale the context length (e.g., 8 = 8x longer context)
    low_freq_factor: float
    high_freq_factor: float
    original_context_len: int  # Original context length model was trained on


class RoPEConfig(base.BaseConfig):
    """
    Configuration for Rotary Position Embeddings.

    Usage patterns:
    1. Standard RoPE: No scaling, full head_dim (base models)
    2. Extended RoPE: With scaling (Llama 3.1 extended context)
    3. Partial RoPE: Only on subset of dims (DeepSeek3 MLA)
    """

    theta: float
    dim: typing.Optional[int] = (
        None  # If None, use full head_dim. If set, use partial RoPE (DeepSeek)
    )

    scaling: typing.Optional[RoPEScalingConfig] = None

    @property
    def use_scaling(self) -> bool:
        """Whether RoPE scaling is enabled."""
        return self.scaling is not None
