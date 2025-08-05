# Standard Library
import abc
import typing

# Third Party
import torch
import torch.nn as nn


class BaseTorchModule(nn.Module, abc.ABC):
    """
    Base class for all our modules that adds common functionality.
    """

    def _maybe_apply_dropout(
        self,
        hidden_states: torch.Tensor,
        dropout_module: typing.Optional[nn.Dropout] = None,
    ) -> torch.Tensor:
        """
        Apply dropout if a dropout module is provided.

        Args:
            hidden_states: Input tensor to potentially apply dropout to
            dropout_module: Optional dropout module. If None, returns input unchanged.

        Returns:
            The input tensor, with dropout applied if dropout_module is not None
        """
        if dropout_module is None:
            return hidden_states
        return dropout_module(hidden_states)
