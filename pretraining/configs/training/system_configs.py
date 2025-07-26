# Standard Library
import typing

# Project
from pretraining.configs import base


class TorchCompilationConfig(base.BaseConfig):
    """Configuration for model compilation/optimization."""

    compile: bool  # PyTorch 2.0 compile
    compile_mode: typing.Optional[typing.Literal["default", "reduce-overhead", "max-autotune"]] = (
        None
    )
