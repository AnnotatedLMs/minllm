# Standard Library
import typing

# Third Party
import jaxtyping
import pydantic
import torch


class TrainingOutput(pydantic.BaseModel):
    """Output from training_forward."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    loss: torch.Tensor
    mtp_losses: typing.Optional[typing.List[torch.Tensor]] = None
    aux_losses: typing.Optional[typing.List[torch.Tensor]] = None


class InferenceOutput(pydantic.BaseModel):
    """Output from inference_forward."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"]
