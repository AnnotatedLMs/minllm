# Standard Library
import dataclasses
import typing


@dataclasses.dataclass
class InitializationConfig:
    """Configuration for model weight initialization."""

    # Initialization strategy
    strategy: typing.Literal["gpt2", "pytorch_default"]

    # GPT-2 specific parameters (only used if strategy="gpt2")
    std: typing.Optional[float] = None
    residual_pattern: typing.Optional[str] = None
    position_init_std: typing.Optional[float] = None

    def __post_init__(self):
        """Validate configuration based on strategy."""
        if self.strategy == "gpt2":
            if self.std is None:
                raise ValueError("std is required for gpt2 strategy")
            if self.residual_pattern is None:
                raise ValueError("residual_pattern is required for gpt2 strategy")
            if self.position_init_std is None:
                raise ValueError("position_init_std is required for gpt2 strategy")
        elif self.strategy == "pytorch_default":
            # No additional parameters needed
            pass
        else:
            raise ValueError(f"Unknown initialization strategy: {self.strategy}")
