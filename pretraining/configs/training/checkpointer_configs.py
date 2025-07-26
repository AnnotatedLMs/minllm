# Standard Library
import typing

# Third Party
import pydantic

# Project
from pretraining.configs import base


class CheckpointerConfig(base.BaseConfig):
    """Configuration for model checkpointing."""

    save_dir: str
    save_interval: int = pydantic.Field(gt=0)
    save_best: bool
    keep_last_n: int = pydantic.Field(ge=0)
    resume_from: typing.Optional[str] = None  # Path to checkpoint to resume from
