# Standard Library
import dataclasses
import typing


@dataclasses.dataclass
class NormalizationConfig:
    """Configuration for normalization layers."""

    norm_type: typing.Literal["layernorm", "rmsnorm"]
    norm_eps: float
    bias: bool  # Whether to use bias in LayerNorm (only for layernorm)
