# Standard Library
import dataclasses
import typing


@dataclasses.dataclass
class FFNConfig:
    """Configuration for feedforward network architecture."""

    # FFN type selection
    ffn_type: typing.Literal["mlp", "swiglu"]

    # Activation function
    activation: typing.Literal["gelu", "relu", "silu", "swish"]

    # Other parameters (required fields)
    dropout: float
    bias: bool

    # Dimension configuration (optional fields)
    intermediate_dim: typing.Optional[int] = None  # If None, computed from expansion_factor
    expansion_factor: typing.Optional[float] = None  # Used if intermediate_dim is None

    # SwiGLU specific (optional fields)
    ffn_dim_multiplier: typing.Optional[float] = None  # Llama-style dimension scaling
    multiple_of: typing.Optional[int] = None  # Round intermediate_dim to multiple of this
