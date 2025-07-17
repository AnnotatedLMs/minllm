# Standard Library
import dataclasses
import typing


@dataclasses.dataclass
class LearningRateScheduleConfig:
    """Configuration for learning rate scheduling."""

    schedule_type: typing.Literal["constant", "cosine", "cosine_with_warmup", "multiphase"]

    # For cosine_with_warmup (most common)
    warmup_iters: int
    lr_decay_iters: int  # Total training steps
    min_lr: float

    # For cosine schedules
    num_cycles: typing.Optional[float] = None  # 0.5 = half cosine (most common)

    # For multiphase (DeepSeek-V3 style)
    phase_steps: typing.Optional[typing.List[int]] = None
    phase_names: typing.Optional[typing.List[str]] = None  # e.g. ['warmup', 'constant', 'decay']

    # Control whether to decay LR at all
    decay_lr: typing.Optional[bool] = None

    def __post_init__(self):
        """Validate LR schedule configuration."""
        if self.warmup_iters < 0:
            raise ValueError(f"warmup_iters must be non-negative, got {self.warmup_iters}")
        if self.lr_decay_iters < self.warmup_iters:
            raise ValueError(
                f"lr_decay_iters ({self.lr_decay_iters}) must be >= warmup_iters ({self.warmup_iters})"
            )
        if self.min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {self.min_lr}")
