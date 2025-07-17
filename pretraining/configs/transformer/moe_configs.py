# Standard Library
import dataclasses
import typing


@dataclasses.dataclass
class MoEConfig:
    """Configuration for Mixture of Experts architecture."""

    # Core MoE parameters
    num_experts: int
    num_experts_per_token: int  # top-k experts per token

    # Expert configuration
    expert_intermediate_dim: typing.Optional[int] = None  # If None, uses FFNConfig

    # MoE variant selection
    moe_type: typing.Literal["standard", "aux_loss_free"] = "standard"

    # Aux-loss-free MoE specific (DeepSeek-V3)
    shared_expert_ratio: float = 0.1  # Fraction of capacity for shared expert
    bias_update_speed: float = 0.001  # Speed of load balancing bias updates

    # Gating configuration
    gate_noise_scale: float = 0.01  # Exploration noise during training
