# Third Party
import pydantic

# Project
from pretraining.configs import base


class MoETrainingConfig(base.BaseConfig):
    """Training configuration specific to MoE models."""

    aux_loss_weight: float = pydantic.Field(default=0.001, ge=0)  # Load balancing loss weight
    capacity_factor: float = pydantic.Field(default=1.25, gt=0)  # Token capacity per expert
    drop_tokens: bool = True  # Drop tokens when experts full
    z_loss_weight: float = pydantic.Field(default=0.001, ge=0)  # Router z-loss for stability


class MultiTokenPredictionTrainingConfig(base.BaseConfig):
    """Training configuration for multi-token prediction."""

    mtp_loss_weight: float = pydantic.Field(
        default=0.1, ge=0
    )  # Weight for multi-token prediction loss
