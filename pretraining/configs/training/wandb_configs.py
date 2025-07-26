# Standard Library
import typing

# Project
from pretraining.configs import base


class WandbConfig(base.BaseConfig):
    """Configuration for Weights & Biases experiment tracking."""

    enabled: bool  # Whether to use W&B logging
    project: str  # W&B project name
    log_model: bool  # Whether to log model checkpoints to W&B
    run_name: typing.Optional[str] = None  # Optional run name
    tags: typing.Optional[typing.List[str]] = None  # Optional tags
