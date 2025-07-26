# Standard Library
import typing

# Project
from pretraining.configs import base


class FFNConfig(base.BaseConfig):
    """Configuration for feedforward network architecture."""

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


class MoEConfig(base.BaseConfig):
    """Configuration for Mixture of Experts architecture."""

    # Core MoE parameters
    num_experts: int
    num_experts_per_token: int  # top-k experts per token

    # Expert configuration
    expert_intermediate_dim: typing.Optional[int] = None  # If None, uses FFNConfig

    # Aux-loss-free MoE specific (DeepSeek-V3)
    shared_expert_ratio: float = 0.1  # Fraction of capacity for shared expert
    bias_update_speed: float = 0.001  # Speed of load balancing bias updates

    # Gating configuration
    gate_noise_scale: float = 0.01  # Exploration noise during training
