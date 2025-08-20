# Standard Library
import typing

# Third Party
import jaxtyping
import pydantic
import torch


class ForwardOutput(pydantic.BaseModel):
    """Unified output from model forward pass."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # Main logits - always present
    logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"]

    # Multi-token prediction logits - only present during training for models with MTP
    mtp_logits: typing.Optional[typing.List[jaxtyping.Float[torch.Tensor, "batch seq vocab"]]] = (
        None
    )

    # Auxiliary losses from MoE layers - only present during training
    aux_losses: typing.Optional[typing.List[torch.Tensor]] = None
