# Standard Library
import typing

# Third Party
import pydantic
import torch


class CheckpointData(pydantic.BaseModel):
    """Data structure for training checkpoints.

    Following OLMo's approach, uses raw state dicts.
    """

    model_config = pydantic.ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # PyTorch state dicts
    model_state: typing.Dict[str, torch.Tensor]
    optimizer_state: typing.Dict[str, typing.Any]
    scheduler_state: typing.Dict[str, typing.Any]

    # Training state (iteration, tokens seen, RNG states, etc.)
    training_state: typing.Dict[str, typing.Any]
