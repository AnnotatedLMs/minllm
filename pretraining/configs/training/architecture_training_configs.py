# Third Party
import pydantic

# Project
from pretraining.configs import base


class MoETrainingConfig(base.BaseConfig):
    """Training configuration specific to MoE models."""

    capacity_factor: float = pydantic.Field(default=1.25, gt=0)  # Token capacity per expert
    drop_tokens: bool = True  # Drop tokens when experts full
