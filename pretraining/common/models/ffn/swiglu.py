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
    Gated feedforward network using element-wise multiplication for adaptive computation.

    Scholarship:
    Quoc et al., 2017, https://arxiv.org/pdf/1710.05941
        - introduces swish activation
    Shazeer, 2020, https://arxiv.org/pdf/2002.05202v1
        - demonstrates swiglu performance in transformers
    Deepseek-V3, 2025, https://arxiv.org/pdf/2412.19437
        - caches SwiGLU inputs and recomputes in backward pass for memory efficiency

    Significance:
    Lets the network learn which features to amplify or suppress through gating.
    More expressive than standard MLP while maintaining similar parameter count.
    Gate values act like attention weights but for individual features rather than tokens.

    Init:
    The three projection layers are defined as:
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)  # Produces gate values
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)    # Produces features
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)  # Projects back
        self.activation = silu (typically)  # Applied to gate only

    Step-by-step control flow (forward):
    1. Receive input of shape [batch, seq_len, hidden_dim]
    2. Project input through gate_proj and apply activation (silu)
    3. Project same input through up_proj (no activation)
    4. Multiply activated gate with up projection element-wise
    5. Project gated features back to hidden_dim through down_proj
    6. Apply dropout if specified
    7. Return output of shape [batch, seq_len, hidden_dim]

    Learning process:
    - Gate projection (self.gate_proj: nn.Linear):
      - Learns to produce gate values between 0 and ~1 (after silu)
      - When model makes mistakes: gradients flow back through multiplication
      - If feature was helpful but gated off: gate weights increase for that pattern
      - If feature was harmful but let through: gate weights decrease for that pattern
      - Result: learns to identify which input patterns should control feature flow

    - Up projection (self.up_proj: nn.Linear):
      - Learns to produce candidate features for gating
      - Gradients modulated by gate values during backprop
      - Features with high gates get stronger gradient signals
      - Features with low gates get weaker gradient signals
      - Result: learns representations that work well with learned gating patterns

    - Down projection (self.down_proj: nn.Linear):
      - Learns to combine gated features back into model dimension
      - Receives gradients from next layer's loss
      - Adjusts to produce useful representations for downstream tasks
      - Result: learns optimal mixing of intermediate features

    - Gating dynamics:
      - Element-wise multiplication creates feature-level attention
      - Smooth gating (via silu) allows gradient flow even for partially closed gates
      - Network learns complementary gate and feature representations
      - Result: adaptive computation that amplifies relevant features per token
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
