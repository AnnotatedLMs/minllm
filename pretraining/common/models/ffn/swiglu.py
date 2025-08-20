# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models.ffn import activation_mixin


class SwiGLU(nn.Module, activation_mixin.ActivationMixin):
    """
    SwiGLU (Swish-Gated Linear Unit) feedforward network.
    https://arxiv.org/pdf/2002.05202v1

    Variation: Gated activation function with learnable gating
    Computation: silu(gate_proj(x)) * up_proj(x) → down_proj
    Effect: More expressive than standard MLP, allows model to selectively amplify or suppress features

    Variation: 2/3 expansion ratio to maintain parameter count
    Computation: When using gating, intermediate_dim = (2/3) * 4 * hidden_dim to match standard MLP params
    Effect: Despite gating overhead, maintains similar parameter budget as standard MLP

    The gating mechanism enables the model to learn which features to emphasize,
    providing a form of adaptive computation that improves model expressiveness.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: typing.Optional[int] = None,
        dropout: typing.Optional[float] = None,
        activation: str = "silu",  # SiLU for SwiGLU by default
        bias: bool = False,  # Llama 3 doesn't use bias
        ffn_dim_multiplier: typing.Optional[float] = None,
        multiple_of: int = 256,
    ):
        super().__init__()

        # Calculate intermediate dimension with Llama-specific logic
        if intermediate_dim is None:
            intermediate_dim = self._calculate_intermediate_dim(
                hidden_dim,
                ffn_dim_multiplier=ffn_dim_multiplier,
                multiple_of=multiple_of,
            )

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # Three projections for gated activation
        # gate_proj: produces gating values with activation
        # up_proj: produces features to be gated (no activation)
        # down_proj: projects back to model dimension
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)

        # Activation for gate (SiLU for SwiGLU, but configurable)
        self.activation = self._get_activation(activation)

        self.ffn_dropout = nn.Dropout(dropout if dropout is not None else 0.0)

    @staticmethod
    def _calculate_intermediate_dim(
        hidden_dim: int,
        ffn_dim_multiplier: typing.Optional[float] = None,
        multiple_of: int = 256,
    ) -> int:
        """
        Calculate intermediate dimension with Llama-specific logic.

        Llama uses 2/3 of the standard 4x expansion to account for the extra
        gate projection while maintaining similar parameter count.
        """
        if ffn_dim_multiplier is not None:
            intermediate_dim = int(hidden_dim * ffn_dim_multiplier)
        else:
            # Llama's specific calculation: 2/3 of 4x expansion
            # Standard MLP: hidden_dim * 4 * hidden_dim * 2 = 8 * hidden_dim^2 params
            # SwiGLU: hidden_dim * intermediate_dim * 3 = 8 * hidden_dim^2 params
            #         when intermediate_dim = 8/3 * hidden_dim
            intermediate_dim = int(2 * hidden_dim * 4 / 3)

        # Round to nearest multiple for hardware efficiency
        intermediate_dim = multiple_of * ((intermediate_dim + multiple_of - 1) // multiple_of)

        return intermediate_dim

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply SwiGLU transformation.

        The SwiGLU process:
        1. Parallel projections - compute both gate and up projections
        2. Gated activation - apply SiLU to gate and multiply with up projection
        3. Output projection - project back to model dimension with dropout

        The gating allows the network to dynamically control information flow,
        learning to emphasize important features while suppressing others.
        """

        # Compute gate with activation
        gate: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        gate = self.activation(self.gate_proj(x))

        # Compute up projection (no activation)
        up: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        up = self.up_proj(x)

        # Apply multiplicative gating
        gated: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        gated = gate * up

        # Project back to hidden dimension
        projected: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        projected = self.down_proj(gated)

        # Apply dropout for regularization
        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self.ffn_dropout(projected)

        return output
