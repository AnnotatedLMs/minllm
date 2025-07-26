# Standard Library
import typing

# Third Party
import pydantic

# Project
from pretraining.configs import base


class OptimizerConfig(base.BaseConfig):
    """Configuration for the optimizer itself."""

    optimizer_type: typing.Literal["adamw"]
    learning_rate: float = pydantic.Field(gt=0, description="Learning rate")
    weight_decay: float = pydantic.Field(ge=0, description="Weight decay coefficient")
    beta1: float = pydantic.Field(ge=0, lt=1, description="Adam beta1")
    beta2: float = pydantic.Field(ge=0, lt=1, description="Adam beta2")
    grad_clip: float = pydantic.Field(gt=0, description="Gradient clipping value")
    eps: float = pydantic.Field(gt=0, description="Adam epsilon")

    # Parameter grouping strategy
    parameter_grouping: typing.Literal["dimension", "name"]
    no_decay_patterns: typing.Optional[typing.List[str]] = None  # For name-based grouping

    @pydantic.model_validator(mode="after")
    def set_default_no_decay_patterns(self):
        """Set default no decay patterns for name-based grouping."""
        if self.parameter_grouping == "name" and self.no_decay_patterns is None:
            self.no_decay_patterns = ["bias", "LayerNorm.weight", "layernorm", "norm"]
        return self
