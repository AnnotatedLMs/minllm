# Standard Library
import typing

# Project
from pretraining.configs import base


class LinearRoPEScalingConfig(base.BaseConfig):
    """
    Configuration for linear RoPE scaling (Llama-style context extension).

    Also known as Position Interpolation (PI) - linearly scales frequencies
    to extend context length while preserving relative position patterns.
    """

    scale_factor: float  # Factor to scale the context length (e.g., 8 = 8x longer context)
    low_freq_factor: float
    high_freq_factor: float
    original_context_len: int  # Original context length model was trained on


class YaRNConfig(base.BaseConfig):
    """
    Configuration for YaRN (Yet another RoPE extension) scaling.

    Used by DeepSeek-V3 for context extension from 4K → 32K → 128K.

    Key differences from standard RoPE scaling:
    - Uses beta parameters instead of frequency factors
    - Includes extrapolation factor for high-frequency dimensions
    - Applies mscale adjustment to attention scaling
    """

    scale_factor: float  # Context extension factor (e.g., 8 for 4K→32K)
    beta_fast: float = 32.0  # High frequency cutoff
    beta_slow: float = 1.0  # Low frequency cutoff
    original_context_len: int = 4096  # Original pretraining context
    extrapolation_factor: float = 1.0  # Factor for extrapolated dimensions
    attn_factor: float = 1.0  # Additional attention scaling factor
    mscale_all_dim: float = 0.1  # Coefficient for mscale computation (0.1 * ln(s) + 1)


class RoPEConfig(base.BaseConfig):
    """
    Configuration for Rotary Position Embeddings.

    Usage patterns:
    1. Standard RoPE: No scaling, full head_dim (base models)
    2. Extended RoPE: With linear scaling (Llama 3.1 extended context)
    3. Partial RoPE: Only on subset of dims (DeepSeek3 MLA)

    Note: YaRN scaling is configured through context extension phases,
    not here, since it's applied dynamically during training.
    """

    theta: float
    dim: typing.Optional[int] = (
        None  # If None, use full head_dim. If set, use partial RoPE (DeepSeek)
    )

    linear_scaling: typing.Optional[LinearRoPEScalingConfig] = None
    """Linear/PI scaling for context extension (Llama-style)."""

    @property
    def use_linear_scaling(self) -> bool:
        """Whether linear RoPE scaling is enabled."""
        return self.linear_scaling is not None
