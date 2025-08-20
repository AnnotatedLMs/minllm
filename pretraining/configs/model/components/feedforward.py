# Standard Library
import typing

# Project
from pretraining.configs import base


class FFNConfig(base.BaseConfig):
    """Configuration for feedforward network architecture."""

    # Activation function
    activation: typing.Literal["gelu", "relu", "silu", "swish"]

    # Other parameters (required fields)
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
    intermediate_dim: typing.Optional[int] = None  # Expert FFN intermediate dimension
    activation: typing.Literal["gelu", "relu", "silu", "swish"] = "silu"  # Expert activation

    # Shared expert configuration (DeepSeek-V3 specific)
    shared_expert_ratio: float = 0.1  # Fraction of capacity for shared expert
    n_shared_experts: int = 2  # Number of shared expert "units" (multiplies intermediate_dim)

    # Load balancing (Aux-loss-free MoE specific)
    bias_update_speed: float = 0.001  # Speed of load balancing bias updates
    aux_loss_alpha: float = 0.001  # Extremely small alpha factor for auxiliary loss (DeepSeek-V3)

    # Gating configuration
    gate_noise_scale: float = 0.01  # Exploration noise during training
