# Standard Library
import dataclasses
import typing


@dataclasses.dataclass
class OptimizerConfig:
    """Configuration for the optimizer itself."""

    optimizer_type: typing.Literal["adamw"]
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    eps: float  # Adam epsilon

    # Parameter grouping strategy
    parameter_grouping: typing.Literal["dimension", "name"]
    no_decay_patterns: typing.Optional[typing.List[str]] = None  # For name-based grouping

    def __post_init__(self):
        """Validate optimizer configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if not 0 <= self.beta1 < 1:
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not 0 <= self.beta2 < 1:
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")
        if self.parameter_grouping == "name" and self.no_decay_patterns is None:
            self.no_decay_patterns = ["bias", "LayerNorm.weight", "layernorm", "norm"]
