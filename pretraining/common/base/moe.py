# Standard Library
import abc
import typing

# Third Party
import torch

# Project
from pretraining.common.base import ffn


class BaseMoE(ffn.BaseFeedForward):
    """
    Pure abstract base class for Mixture of Experts layers.

    MoE layers route inputs to different expert networks, but the specific
    routing mechanism and expert architecture is left to subclasses.
    """

    @abc.abstractmethod
    def get_auxiliary_loss(self) -> typing.Optional[torch.Tensor]:
        """
        Get any auxiliary loss (e.g., load balancing) from the last forward pass.

        Returns:
            Auxiliary loss tensor, or None if not applicable
        """
        pass
