# Standard Library
import typing

# Third Party
import jaxtyping
import pydantic
import torch


class TrainingInputs(pydantic.BaseModel):
    """Inputs for training_forward."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    input_ids: jaxtyping.Int[torch.Tensor, "batch seq"]
    labels: jaxtyping.Int[torch.Tensor, "batch seq"]
    attention_mask: typing.Optional[torch.Tensor] = None
    mtp_targets: typing.Optional[jaxtyping.Int[torch.Tensor, "batch depth seq"]] = None


class InferenceInputs(pydantic.BaseModel):
    """Inputs for inference_forward."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    input_ids: jaxtyping.Int[torch.Tensor, "batch seq"]
    attention_mask: typing.Optional[torch.Tensor] = None
    position_offset: int = 0  # For KV cache in Llama
