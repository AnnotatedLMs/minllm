# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models.ffn import activation_mixin


class MLP(nn.Module, activation_mixin.ActivationMixin):
    """
    Standard MLP (Multi-Layer Perceptron) pattern.

    Used by: GPT-2

    Variation: Simple two-layer design with single activation
    Computation: Linear → Activation → Linear → Dropout
    Effect: Model learns position-wise transformations with smooth non-linearity

    Variation: Fixed 4x expansion ratio
    Computation: Hidden dimension expanded by factor of 4
    Effect: Provides sufficient capacity for feature transformation without excessive parameters
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: typing.Optional[int] = None,
        dropout: typing.Optional[float] = None,
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()

        # Default to 4x expansion (GPT-2 standard)
        if intermediate_dim is None:
            intermediate_dim = 4 * hidden_dim

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # Two-layer MLP projections
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)
        self.activation = self._get_activation(activation)

        self.ffn_dropout = nn.Dropout(dropout if dropout is not None else 0.0)

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Apply standard MLP transformation."""

        intermediate: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        intermediate = self.up_proj(x)

        activated: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        activated = self.activation(intermediate)

        projected: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        projected = self.down_proj(activated)

        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self.ffn_dropout(projected)

        return output
