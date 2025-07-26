# Standard Library
import typing

# Third Party
import pydantic

# Project
from pretraining.configs import base


class LearningRateScheduleConfig(base.BaseConfig):
    """Configuration for learning rate scheduling."""

    schedule_type: typing.Literal["constant", "cosine", "cosine_with_warmup", "multiphase"]

    # For cosine_with_warmup (most common)
    warmup_iters: int = pydantic.Field(ge=0)
    lr_decay_iters: int  # Total training steps
    min_lr: float = pydantic.Field(ge=0)

    # For cosine schedules
    num_cycles: typing.Optional[float] = None  # 0.5 = half cosine (most common)

    # For multiphase (DeepSeek-V3 style)
    phase_steps: typing.Optional[typing.List[int]] = None
    phase_names: typing.Optional[typing.List[str]] = None  # e.g. ['warmup', 'constant', 'decay']

    # Control whether to decay LR at all
    decay_lr: typing.Optional[bool] = None

    @pydantic.model_validator(mode="after")
    def validate_lr_schedule(self):
        """Validate LR schedule configuration."""
        if self.lr_decay_iters < self.warmup_iters:
            raise ValueError(
                f"lr_decay_iters ({self.lr_decay_iters}) must be >= warmup_iters ({self.warmup_iters})"
            )
        return self
