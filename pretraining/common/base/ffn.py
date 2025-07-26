# Standard Library
import abc

# Third Party
import jaxtyping
import torch

# Project
from pretraining.common.base import core


class BaseFeedForward(core.BaseTorchModule, abc.ABC):
    """
    Pure abstract base class for feedforward components.

    This encompasses all types of feedforward layers including:
    - Standard MLPs
    - Gated variants (SwiGLU, GeGLU, etc.)
    - Mixture of Experts
    """

    @abc.abstractmethod
    def forward(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply feedforward transformation.

        Args:
            x: Input tensor of shape [batch, seq, d_model]

        Returns:
            Output tensor of shape [batch, seq, d_model]
        """
        pass
