# Standard Library

# Third Party
import pydantic

# Project
from pretraining.configs import base


class BaseInitializationConfig(base.BaseConfig):
    """Base configuration for model weight initialization."""

    pass


class PyTorchDefaultInitConfig(BaseInitializationConfig):
    """Use PyTorch's default initialization (Kaiming uniform)."""

    pass


class GPT2InitConfig(BaseInitializationConfig):
    """GPT-2 style initialization with specific standard deviations."""

    std: float = pydantic.Field(gt=0, description="Standard deviation for weight initialization")
    residual_pattern: str = pydantic.Field(description="Regex pattern to identify residual layers")
    position_init_std: float = pydantic.Field(
        gt=0, description="Standard deviation for position embeddings"
    )
