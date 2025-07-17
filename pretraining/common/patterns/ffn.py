# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
# Local
from pretraining.common.base.models import ffn

# TODO Double check intermediate dimension calculation logic


class FeedForward(ffn.BaseFeedForward):
    """
    Base class for feedforward patterns with common implementations.

    This class provides standard operations shared across feedforward
    networks.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout

        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),  # Also known as swish
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]


class MLP(FeedForward):
    """
    Standard MLP (Multi-Layer Perceptron) pattern.

    Used by: GPT-2.
    Pattern: Linear → Activation → Linear → Dropout

    This is the classic feedforward network with a single activation function
    and 4x hidden dimension expansion.
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
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq intermediate"]:
        """
        Expand input to intermediate dimension.

        This is the first linear transformation that increases dimensionality.
        """
        expanded: jaxtyping.Float[torch.Tensor, "batch seq intermediate"]
        expanded = self.c_fc(x)
        return expanded

    def _apply_activation(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq intermediate"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq intermediate"]:
        """
        Apply activation function.

        GPT-2 uses GELU for smooth, differentiable non-linearity.
        """
        activated: jaxtyping.Float[torch.Tensor, "batch seq intermediate"]
        activated = self.activation(x)
        return activated

    def _project_back(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq intermediate"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Project back to model dimension.

        This reduces dimensionality back to the original size.
        """
        projected: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        projected = self.c_proj(x)
        return projected

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply standard MLP transformation.

        The process:
        1. Expand to intermediate dimension (4x hidden_dim)
        2. Apply activation function (GELU for GPT-2)
        3. Project back to model dimension
        4. Apply dropout for regularization
        """

        intermediate: jaxtyping.Float[torch.Tensor, "batch seq intermediate"]
        intermediate = self._expand_to_intermediate(x)

        activated: jaxtyping.Float[torch.Tensor, "batch seq intermediate"]
        activated = self._apply_activation(intermediate)

        projected: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        projected = self._project_back(activated)

        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self.dropout_layer(projected) if self.dropout > 0 else projected

        return output


class MultiplicativeGatedFFN(FeedForward):
    """
    Multiplicative gated feedforward network pattern (SwiGLU family).

    Used by: Llama (SwiGLU variant).
    Pattern: gate(x) * up(x) → down()

    The gating mechanism allows the model to dynamically control information flow.
    The "Swish-Gated Linear Unit" uses SiLU (Swish) as the gating activation.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: typing.Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "silu",  # SiLU for SwiGLU
        bias: bool = False,
        ffn_dim_multiplier: typing.Optional[float] = None,
        multiple_of: int = 256,
    ):
        if intermediate_dim is None:
            intermediate_dim = self.calculate_intermediate_dim(
                hidden_dim,
                expansion_factor=4.0,  # Llama's specific expansion factor
                ffn_dim_multiplier=ffn_dim_multiplier,
                multiple_of=multiple_of,
            )

        super().__init__(hidden_dim, intermediate_dim, dropout)

        # Three projections for gated FFN
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)

        # Activation for gate
        self.activation = self._get_activation(activation)

    @staticmethod
    def calculate_intermediate_dim(
        hidden_dim: int,
        expansion_factor: float = 4.0,
        ffn_dim_multiplier: typing.Optional[float] = None,
        multiple_of: int = 256,
    ) -> int:
        """
        Calculate intermediate dimension with model-specific logic.

        Used by Llama/SwiGLU variants for optimal dimension sizing.
        """
        if ffn_dim_multiplier is not None:
            intermediate_dim = int(hidden_dim * ffn_dim_multiplier)
        else:
            # Llama's specific calculation: 2/3 of 4x expansion
            intermediate_dim = int(2 * hidden_dim * expansion_factor / 3)

        # Round to nearest multiple for efficiency
        intermediate_dim = multiple_of * ((intermediate_dim + multiple_of - 1) // multiple_of)

        return intermediate_dim

    def _compute_gate(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq intermediate"]:
        """
        Compute gating values with activation.

        The gate controls how much information flows through.
        """
        # Linear projection
        gate_pre_activation: jaxtyping.Float[torch.Tensor, "batch seq intermediate"]
        gate_pre_activation = self.gate_proj(x)

        # Apply activation (SiLU for SwiGLU)
        gate: jaxtyping.Float[torch.Tensor, "batch seq intermediate"]
        gate = self.activation(gate_pre_activation)

        return gate

    def _compute_up_projection(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq intermediate"]:
        """
        Compute up projection (no activation).

        This is the actual information to be gated.
        """
        up: jaxtyping.Float[torch.Tensor, "batch seq intermediate"]
        up = self.up_proj(x)
        return up

    def _apply_gating(
        self,
        gate: jaxtyping.Float[torch.Tensor, "batch seq intermediate"],
        up: jaxtyping.Float[torch.Tensor, "batch seq intermediate"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq intermediate"]:
        """
        Apply multiplicative gating.

        Element-wise multiplication allows the gate to control
        information flow dynamically.
        """
        gated: jaxtyping.Float[torch.Tensor, "batch seq intermediate"]
        gated = gate * up
        return gated

    def _compute_down_projection(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq intermediate"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Project back to model dimension."""
        down: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        down = self.down_proj(x)
        return down

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply SwiGLU-style gated feedforward.

        The process:
        1. Compute gate with activation (SiLU for SwiGLU)
        2. Compute up projection (no activation)
        3. Multiply gate and up projection element-wise
        4. Project back to model dimension
        5. Apply dropout if specified

        The gating allows the model to learn to selectively pass information.
        """
        gate: jaxtyping.Float[torch.Tensor, "batch seq intermediate"]
        gate = self._compute_gate(x)

        up: jaxtyping.Float[torch.Tensor, "batch seq intermediate"]
        up = self._compute_up_projection(x)

        gated: jaxtyping.Float[torch.Tensor, "batch seq intermediate"]
        gated = self._apply_gating(gate, up)

        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self._compute_down_projection(gated)

        output = self.dropout_layer(output) if self.dropout > 0 else output

        return output
