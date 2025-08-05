# Standard Library
import typing

# Third Party
import pydantic

# Project
from pretraining.configs import base


class LossConfig(base.BaseConfig):
    """Configuration for loss computation and weighting.

    Controls how different loss components are weighted during training.
    Only include fields relevant to your model architecture.
    """

    # Cross-entropy loss for next token prediction (always required)
    cross_entropy_weight: float = pydantic.Field(
        default=1.0, ge=0, description="Weight for cross-entropy loss (typically always 1.0)"
    )

    # Optional: MoE auxiliary loss for load balancing (only for MoE models)
    moe_aux_loss_weight: typing.Optional[float] = pydantic.Field(
        default=None, ge=0, description="Weight for MoE load balancing loss (only for MoE models)"
    )

    # Optional: Multi-token prediction loss weight (only for MTP models)
    mtp_loss_weight: typing.Optional[float] = pydantic.Field(
        default=None, ge=0, description="Weight λ for averaged MTP loss (only for models with MTP)"
    )

    # Optional: Z-loss for training stability (from OLMo)
    z_loss_weight: typing.Optional[float] = pydantic.Field(
        default=None, ge=0, description="Weight for z-loss (logit regularization)"
    )

    # Loss computation options
    ignore_index: int = pydantic.Field(
        default=-100, description="Token ID to ignore in loss computation"
    )

    # Optional: fused loss for performance (requires flash-attn)
    use_fused_loss: bool = pydantic.Field(
        default=False, description="Use fused cross-entropy from flash-attn if available"
    )
