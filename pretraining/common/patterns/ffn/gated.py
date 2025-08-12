# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.patterns.ffn import core


class MultiplicativeGatedFFN(core.FeedForward):
    """
    Multiplicative gated feedforward network pattern (SwiGLU family).

    Used by: Llama (SwiGLU variant)

    Variation: Multiplicative gating with separate gate and up projections
    Computation: gate(x) * up(x) → down()
    Effect: Model learns to selectively filter information based on input content

    Variation: Uses SiLU (Swish) activation for gating
    Computation: Smooth gating function that can partially close/open
    Effect: More nuanced control over information flow compared to binary gates
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
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]:
        """
        Compute gating values with activation.

        The gate controls how much information flows through,
        learning input-dependent filtering.
        """
        # Linear projection
        gate_pre_activation: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        gate_pre_activation = self.gate_proj(x)

        # Apply activation (SiLU for SwiGLU)
        gate: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        gate = self.activation(gate_pre_activation)

        return gate

    def _compute_up_projection(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]:
        """
        Compute up projection (no activation).

        This is the actual information to be gated,
        representing the features to be selectively passed.
        """
        up: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        up = self.up_proj(x)
        return up

    def _apply_gating(
        self,
        gate: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"],
        up: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]:
        """
        Apply multiplicative gating.

        Element-wise multiplication allows the gate to control
        information flow dynamically, learning which features to emphasize.
        """
        gated: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        gated = gate * up
        return gated

    def _compute_down_projection(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        down: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        down = self.down_proj(x)
        return down

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply SwiGLU-style gated feedforward.

        The process:
        1. Compute gate with activation (SiLU for SwiGLU) - learns input-dependent filtering
        2. Compute up projection (no activation) - transforms features for gating
        3. Multiply gate and up projection element-wise - applies learned filtering
        4. Project back to model dimension - compresses filtered features
        5. Apply dropout if specified - regularizes during training

        The gating allows the model to dynamically control information flow based on content.
        """
        gate: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        gate = self._compute_gate(x)

        up: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        up = self._compute_up_projection(x)

        gated: jaxtyping.Float[torch.Tensor, "batch seq_len intermediate"]
        gated = self._apply_gating(gate, up)

        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self._compute_down_projection(gated)

        output = self._maybe_apply_dropout(output, self.ffn_dropout)

        return output
