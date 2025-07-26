# Third Party
import pydantic

# Project
from pretraining.configs import base

# TODO: adjust based on dataset/ dataloader requirements


class DataConfig(base.BaseConfig):
    """Configuration for data loading."""

    dataset: str
    data_dir: str
    num_workers: int = pydantic.Field(ge=0, description="Number of data loading workers")
    pin_memory: bool = pydantic.Field(
        default=True, description="Pin memory for faster GPU transfer"
    )
