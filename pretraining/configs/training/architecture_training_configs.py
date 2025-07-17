# Standard Library
import dataclasses


@dataclasses.dataclass
class MoETrainingConfig:
    """Training configuration specific to MoE models."""

    aux_loss_weight: float = 0.001  # Load balancing loss weight
    capacity_factor: float = 1.25  # Token capacity per expert
    drop_tokens: bool = True  # Drop tokens when experts full
    z_loss_weight: float = 0.001  # Router z-loss for stability


@dataclasses.dataclass
class MultiTokenPredictionTrainingConfig:
    """Training configuration for multi-token prediction."""

    mtp_loss_weight: float = 0.1  # Weight for multi-token prediction loss
