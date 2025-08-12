# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.patterns.attention import core


class MultiHeadAttention(core.Attention):
    """
    Multi-Head Attention (MHA) - The classic attention pattern.

    Used by: GPT-2

    Variation: Every attention head gets its own key and value vectors
    Computation: Uses a combined QKV projection for efficiency
    Effect: Each head can specialize in different types of relationships (syntax, semantics, etc.)

    Variation: Relies on learned position embeddings in the input
    Computation: Position information added before attention (not during)
    Effect: Model learns fixed positional patterns up to max training length

    See forward() for the specific flow: projection → split → attention → merge → output
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: typing.Optional[float] = None,
        bias: bool = True,
        max_seq_length: int = 1024,
        is_causal: bool = True,
        use_flash_attention: bool = True,
    ):
        super().__init__(hidden_dim, num_heads, dropout, is_causal, use_flash_attention)

        self.max_seq_length = max_seq_length

        # Combined QKV projection
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.resid_dropout = nn.Dropout(dropout) if dropout is not None else None

        # Only create bias buffer if we're not using Flash Attention
        # (Flash handles causal masking internally)
        if not self.use_flash_attention:
            # Buffer: fixed causal mask that prevents attention to future tokens
            # Not a learned parameter - just a precomputed mask for efficiency
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(max_seq_length, max_seq_length)).view(
                    1, 1, max_seq_length, max_seq_length
                ),
            )

    def _compute_qkv_projections(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ]:
        """
        Compute Q, K, V projections using combined projection matrix.

        GPT-2 specific: Single linear layer produces all three projections.
        """
        # Combined projection to 3x hidden dimension
        qkv: jaxtyping.Float[torch.Tensor, "batch seq_len 3*hidden_dim"] = self.c_attn(x)

        # Split into Q, K, V
        q: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        k: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        v: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        q, k, v = qkv.split(self.hidden_dim, dim=2)

        return q, k, v

    def _reshape_to_multihead(
        self,
        tensor: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        batch_size: int,
        seq_len: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]:
        """Reshape tensor for multi-head attention computation."""
        # First reshape to separate head dimension
        reshaped: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads head_dim"]
        reshaped = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to put heads before sequence
        multihead: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        multihead = reshaped.transpose(1, 2)

        return multihead

    def _apply_causal_mask(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
        seq_len: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """
        Apply causal mask to attention scores.

        Override: GPT-2 uses pre-computed buffer for efficiency.
        """
        if self.use_flash_attention:
            # Flash attention handles causal masking internally
            return scores

        # Use pre-computed mask
        masked_scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        masked_scores = scores.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        return masked_scores

    def _merge_heads(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Merge attention heads back to single tensor."""
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Transpose heads and sequence dimensions
        x_transposed: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads head_dim"]
        x_transposed = x.transpose(1, 2).contiguous()

        # Reshape to combine heads
        x_merged: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        x_merged = x_transposed.view(batch_size, seq_len, self.hidden_dim)

        return x_merged

    def _apply_output_projection(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Apply output projection."""
        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self.c_proj(x)
        return output

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply standard multi-head attention.

        The MHA process:
        1. Combined projection - single matrix multiply produces Q, K, V efficiently
        2. Split and reshape - separate Q, K, V and organize into attention heads
        3. Compute attention - compute scaled dot-product attention for each head
        4. Merge heads - concatenate all head outputs back together
        5. Output projection - final linear transform with dropout

        Note: Position information comes from learned embeddings added before this layer.
        """

        batch_size, seq_len, hidden_dim = x.shape

        q: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        k: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        v: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        q, k, v = self._compute_qkv_projections(x)

        q_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        k_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        v_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        q_heads = self._reshape_to_multihead(q, batch_size, seq_len)
        k_heads = self._reshape_to_multihead(k, batch_size, seq_len)
        v_heads = self._reshape_to_multihead(v, batch_size, seq_len)

        # Note: MultiHeadAttention does not use KV caching or position_offset
        # GPT-2 uses learned position embeddings added to input, not RoPE

        attn_output: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        if self.use_flash_attention:
            attn_output = self._apply_flash_attention(q_heads, k_heads, v_heads, attention_mask)
        else:
            attn_output = self._compute_manual_attention(q_heads, k_heads, v_heads, attention_mask)

        merged: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        merged = self._merge_heads(attn_output)

        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self._apply_output_projection(merged)

        output = self._maybe_apply_dropout(output, self.resid_dropout)

        return output
