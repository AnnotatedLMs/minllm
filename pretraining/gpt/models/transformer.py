# Standard Library
import logging

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from pretraining.gpt.models import attention
from pretraining.gpt.models import config
from pretraining.gpt.models import mlp
from pretraining.gpt.models import norm

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers."""

    def __init__(self, config: config.GPTConfig):
        super().__init__()
        self.ln_1 = norm.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = attention.CausalSelfAttention(config)
        self.ln_2 = norm.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = mlp.MLP(config)

    def _apply_attention_with_residual(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Apply attention layer with residual connection."""
        # Normalize, apply attention, add residual
        return x + self.attn(self.ln_1(x))

    def _apply_mlp_with_residual(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Apply MLP layer with residual connection."""
        # Normalize, apply MLP, add residual
        return x + self.mlp(self.ln_2(x))

    def forward(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply transformer block.

        The process:
        1. Apply attention with residual connection
        2. Apply MLP with residual connection
        """
        x = self._apply_attention_with_residual(x)
        x = self._apply_mlp_with_residual(x)
        return x
