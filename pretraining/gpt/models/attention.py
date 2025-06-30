# Standard Library
import logging
import math
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn
from torch.nn import functional as F

# Project
from pretraining.gpt.models import config

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer."""

    def __init__(self, config: config.GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention support (PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            logger.warning("Using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask to ensure attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def _extract_dimensions(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> typing.Tuple[int, int, int]:
        """Extract batch size, sequence length, and embedding dimension."""
        B, T, C = x.size()
        return B, T, C

    def _compute_qkv_projections(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input to query, key, and value representations."""
        qkv: jaxtyping.Float[torch.Tensor, "batch seq 3*d_model"] = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        return q, k, v

    def _reshape_for_multihead_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, B: int, T: int, C: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape tensors for multi-head attention computation."""
        head_dim = C // self.n_head

        # Reshape and transpose to [batch, n_heads, seq, head_dim]
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        return q, k, v

    def _apply_causal_attention(
        self,
        q: jaxtyping.Float[torch.Tensor, "batch n_head seq head_dim"],
        k: jaxtyping.Float[torch.Tensor, "batch n_head seq head_dim"],
        v: jaxtyping.Float[torch.Tensor, "batch n_head seq head_dim"],
        T: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_head seq head_dim"]:
        """Apply causal self-attention mechanism."""
        if self.flash:
            # Use optimized Flash Attention
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Manual attention implementation
            return self._manual_attention(q, k, v, T)

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, T: int
    ) -> torch.Tensor:
        """Compute attention manually when Flash Attention is not available."""
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # Convert to probabilities
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        return att @ v

    def _merge_heads(
        self, y: jaxtyping.Float[torch.Tensor, "batch n_head seq head_dim"], B: int, T: int, C: int
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Merge attention heads back into single tensor."""
        return y.transpose(1, 2).contiguous().view(B, T, C)

    def _apply_output_projection(
        self, y: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Apply final output projection and dropout."""
        return self.resid_dropout(self.c_proj(y))

    def forward(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply multi-head causal self-attention.

        The process:
        1. Extract dimensions from input
        2. Compute query, key, value projections
        3. Reshape for multi-head processing
        4. Apply causal attention mechanism
        5. Merge heads back together
        6. Apply output projection
        """
        # Step 1: Extract dimensions
        B, T, C = self._extract_dimensions(x)

        # Step 2: Compute Q, K, V projections
        q, k, v = self._compute_qkv_projections(x)

        # Step 3: Reshape for multi-head attention
        q, k, v = self._reshape_for_multihead_attention(q, k, v, B, T, C)

        # Step 4: Apply causal self-attention
        y = self._apply_causal_attention(q, k, v, T)

        # Step 5: Merge heads
        y = self._merge_heads(y, B, T, C)

        # Step 6: Apply output projection
        y = self._apply_output_projection(y)

        return y
