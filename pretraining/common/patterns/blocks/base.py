# Standard Library

# Third Party
import jaxtyping
import torch

# Project
from pretraining.common.base import transformer


class TransformerBlock(transformer.BaseTransformerBlock):
    """
    Educational base class showing common transformer block patterns.

    Most transformer blocks share the same core operations:
    1. Apply normalization - stabilizes training by controlling activation magnitudes
    2. Apply attention sublayer - enables tokens to exchange information
    3. Apply residual connection - preserves gradient flow and enables feature reuse
    4. Apply normalization again - prepares features for next transformation
    5. Apply feedforward sublayer - applies position-wise transformations
    6. Apply residual connection again - maintains information flow

    The variations between architectures typically involve:
    - Type of normalization (LayerNorm vs RMSNorm)
    - Presence of bias terms
    - Type of attention mechanism (MHA vs GQA vs MLA)
    - Type of feedforward network (MLP vs Gated vs MoE)

    See the forward() method in each implementation for the specific flow.
    """

    def __init__(self):
        super().__init__()

    def _apply_residual_connection(
        self,
        residual: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Apply residual connection - preserves information and gradients across layers."""
        combined: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        combined = residual + output
        return combined
