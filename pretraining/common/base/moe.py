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
    def _create_expert(self, hidden_dim: int, intermediate_dim: int) -> torch.nn.Module:
        """
        Create a single expert network.

        Each MoE implementation should define its own expert architecture.

        Args:
            hidden_dim: Input/output dimension of the expert
            intermediate_dim: Hidden dimension within the expert

        Returns:
            nn.Module: The expert network
        """
        pass

    @abc.abstractmethod
    def get_auxiliary_loss(self) -> typing.Optional[torch.Tensor]:
        """
        Get any auxiliary loss (e.g., load balancing) from the last forward pass.

        Returns:
            Auxiliary loss tensor, or None if not applicable
        """
        pass
