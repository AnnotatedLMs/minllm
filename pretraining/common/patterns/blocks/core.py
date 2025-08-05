# Standard Library
import typing

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
        residual: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Apply residual connection - preserves information and gradients across layers."""
        combined: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        combined = residual + output
        return combined


class PrenormTransformerBlock(TransformerBlock):
    """
    Base class for pre-normalization transformer blocks.

    Pre-normalization is the dominant pattern in modern transformers where
    normalization is applied BEFORE each sublayer rather than after.

    The pattern:
    1. Normalize → Attention → Residual
    2. Normalize → FFN → Residual

    This provides better training stability for deep networks and is used by:
    - GPT-2 (with LayerNorm)
    - Llama (with RMSNorm)
    - DeepSeek (with RMSNorm)

    Subclasses implement:
    - Specific normalization type (LayerNorm vs RMSNorm)
    - Specific attention mechanism (MHA vs GQA vs MLA)
    - Specific FFN type (MLP vs Gated vs MoE)
    """

    def __init__(self):
        super().__init__()
        # Subclasses will initialize:
        # - self.attention_norm: normalization before attention
        # - self.ffn_norm: normalization before FFN
        # - self.attention: attention module
        # - self.ffn: feedforward module (or self.moe for MoE variants)

    def _apply_attention_sublayer(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply pre-norm attention sublayer.

        Pattern: Normalize → Attention → Residual
        This stabilizes training by ensuring attention receives normalized inputs.
        """
        # Save input for residual
        residual = x

        # Apply normalization
        x = self.attention_norm(x)

        # Apply attention
        x = self.attention(x, attention_mask=attention_mask, **kwargs)

        # Apply residual connection
        return self._apply_residual_connection(residual, x)

    def _apply_ffn_sublayer(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply pre-norm FFN sublayer.

        Pattern: Normalize → FFN → Residual
        Position-wise transformations with normalized inputs.
        """
        # Save input for residual
        residual = x

        # Apply normalization
        x = self.ffn_norm(x)

        # Apply FFN (or MoE)
        if hasattr(self, "ffn"):
            x = self.ffn(x)
        elif hasattr(self, "moe"):
            x = self.moe(x)
        else:
            raise AttributeError(
                f"{self.__class__.__name__} has no feedforward component (ffn or moe)"
            )

        # Apply residual connection
        return self._apply_residual_connection(residual, x)

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply pre-norm transformer block.

        The standard two-sublayer pattern:
        1. Attention sublayer with pre-norm
        2. FFN sublayer with pre-norm

        Additional kwargs are passed to attention (e.g., position_offset for KV caching).
        """
        # Apply attention sublayer
        x = self._apply_attention_sublayer(x, attention_mask=attention_mask, **kwargs)

        # Apply FFN sublayer
        x = self._apply_ffn_sublayer(x)

        return x
