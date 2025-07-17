# Standard Library
import abc
import typing

# Third Party
import jaxtyping
import torch

# Project
# Local
from pretraining.common.base.models import core


class BaseTransformerBlock(core.BaseTorchModule, abc.ABC):
    """
    Pure abstract base class for transformer blocks.

    A transformer block combines attention and feedforward layers,
    but the specific arrangement (pre-norm, post-norm, parallel, etc.)
    is left entirely to subclasses.
    """

    @abc.abstractmethod
    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        past_kv_cache: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Apply transformer block.

        Args:
            x: Input hidden states
            attention_mask: Optional attention mask
            past_kv_cache: Optional past key-value cache for this layer
            use_cache: Whether to return updated cache

        Returns:
            Tuple of:
            - Output hidden states
            - Optional updated cache
        """
        pass
