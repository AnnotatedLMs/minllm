# Standard Library
import abc
import typing

# Third Party
import jaxtyping
import torch

# Project
# Local
from pretraining.common.base.models import core


class BaseAttention(core.BaseTorchModule, abc.ABC):
    """
    Pure abstract base class for attention mechanisms.

    This defines the interface that all attention implementations must follow,
    without prescribing any specific implementation details.
    """

    @abc.abstractmethod
    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply attention mechanism.

        Args:
            x: Input tensor of shape [batch, seq, d_model]
            attention_mask: Optional mask for attention scores
            position_offset: Starting position for RoPE (used in autoregressive generation)
            **kwargs: Additional architecture-specific arguments

        Returns:
            Output tensor of shape [batch, seq, d_model]
        """
        pass
