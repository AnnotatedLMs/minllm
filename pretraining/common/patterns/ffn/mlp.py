# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from pretraining.common.patterns.ffn import core


class MLP(core.FeedForward):
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
        dropout: float = 0.0,
        activation: str = "gelu",
        bias: bool = True,
    ):
        # Default to 4x expansion (GPT-2 standard)
        if intermediate_dim is None:
            intermediate_dim = 4 * hidden_dim

        super().__init__(hidden_dim, intermediate_dim, dropout)

        # Layers
        self.c_fc = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.c_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)

        # Activation
        self.activation = self._get_activation(activation)

    def _expand_to_intermediate(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]:
        """
        Expand input to intermediate dimension.

        This is the first linear transformation that increases dimensionality,
        giving the model more capacity to learn complex features.
        """
        expanded: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        expanded = self.c_fc(x)
        return expanded

    def _apply_activation(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]:
        """
        Apply activation function.

        GPT-2 uses GELU for smooth, differentiable non-linearity,
        enabling gradient flow while introducing necessary non-linearity.
        """
        activated: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        activated = self.activation(x)
        return activated

    def _project_back(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Project back to model dimension.

        This reduces dimensionality back to the original size,
        forcing the model to compress learned features.
        """
        projected: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        projected = self.c_proj(x)
        return projected

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply standard MLP transformation.

        The process:
        1. Expand to intermediate dimension (4x hidden_dim) - increases capacity for feature learning
        2. Apply activation function (GELU for GPT-2) - introduces non-linearity for complex transformations
        3. Project back to model dimension - compresses features to original size
        4. Apply dropout for regularization - improves generalization during training
        """

        intermediate: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        intermediate = self._expand_to_intermediate(x)

        activated: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        activated = self._apply_activation(intermediate)

        projected: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        projected = self._project_back(activated)

        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self._maybe_apply_dropout(projected, self.ffn_dropout)

        return output
