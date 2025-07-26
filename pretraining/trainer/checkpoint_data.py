# Standard Library
import typing

# Third Party
import pydantic
import torch

# Project
from pretraining.configs import base
from pretraining.configs.training import trainer_configs


class CheckpointData(base.BaseConfig):
    """Data structure for training checkpoints.

    Organizes all checkpoint components in a clear structure
    instead of using untyped dictionaries.
    """

    # Override the base config to allow arbitrary types (needed for torch tensors)
    model_config = pydantic.ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # PyTorch state dicts
    model_state_dict: typing.Dict[str, torch.Tensor]
    optimizer_state_dict: typing.Dict[str, typing.Any]
    scheduler_state_dict: typing.Dict[str, typing.Any]
    scaler_state_dict: typing.Optional[typing.Dict[str, typing.Any]] = None

    # Training progress (from TrainingState)
    training_state_dict: typing.Dict[str, typing.Any]  # From state.get_checkpoint_dict()

    # Data loading state
    dataloader_state_dict: typing.Optional[typing.Dict[str, typing.Any]] = None

    # Configuration
    config: trainer_configs.TrainingLoopConfig
    model_args: typing.Optional[typing.Dict[str, typing.Any]] = None

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        """Convert to dictionary for saving."""
        return {
            "model": self.model_state_dict,
            "optimizer": self.optimizer_state_dict,
            "scheduler": self.scheduler_state_dict,
            "scaler": self.scaler_state_dict,
            "training_state": self.training_state_dict,
            "dataloader": self.dataloader_state_dict,
            "config": self.config.model_dump(),  # Convert Pydantic model to dict
            "model_args": self.model_args,
        }

    @classmethod
    def from_dict(cls, data: typing.Dict[str, typing.Any]) -> "CheckpointData":
        """Create from dictionary loaded from checkpoint."""
        # Handle config - it might be a dict or already a TrainingLoopConfig
        config = data.get("config")
        if isinstance(config, dict):
            config = trainer_configs.TrainingLoopConfig.model_validate(config)

        return cls(
            model_state_dict=data["model"],
            optimizer_state_dict=data["optimizer"],
            scheduler_state_dict=data["scheduler"],
            scaler_state_dict=data.get("scaler"),
            training_state_dict=data.get("training_state", {}),
            dataloader_state_dict=data.get("dataloader"),
            config=config,
            model_args=data.get("model_args"),
        )
