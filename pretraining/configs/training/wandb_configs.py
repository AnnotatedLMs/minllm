# Standard Library
import typing

# Third Party
import pydantic

# Project
from pretraining.configs import base


class WandbConfig(base.BaseConfig):
    """Configuration for Weights & Biases experiment tracking."""

    enabled: bool = pydantic.Field(True, description="Whether to use W&B logging")
    project: typing.Optional[str] = pydantic.Field(None, description="W&B project name")
    entity: typing.Optional[str] = pydantic.Field(None, description="W&B entity (team/username)")
    group: typing.Optional[str] = pydantic.Field(None, description="Group runs together")
    name: typing.Optional[str] = pydantic.Field(
        None, description="Run name (defaults to random name)"
    )
    tags: typing.Optional[typing.List[str]] = pydantic.Field(
        default_factory=list, description="Tags for this run"
    )
    log_artifacts: bool = pydantic.Field(
        False, description="Whether to log artifacts (checkpoints, etc.) to W&B"
    )
    rank_zero_only: bool = pydantic.Field(
        True, description="Only log from rank 0 in distributed training"
    )
    log_interval: int = pydantic.Field(1, gt=0, description="How often to log metrics (in steps)")
