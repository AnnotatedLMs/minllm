# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models.attention import attention_mixins
from pretraining.common.models.attention import projection_mixins


class MultiHeadAttention(
    nn.Module,
    projection_mixins.FusedQKVProjectionMixin,
    attention_mixins.MultiHeadReshapeMixin,
    attention_mixins.ManualAttentionMixin,
    attention_mixins.FlashAttentionMixin,
):
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
        attention_scale: typing.Optional[float] = None,
    ):
        super().__init__()

        # Store configuration
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.is_causal = is_causal
        self.use_flash_attention = use_flash_attention
        self.attention_scale = attention_scale
        self.max_seq_length = max_seq_length

        assert hidden_dim % num_heads == 0

        # QKV projection components
        self.qkv_projection = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)

        # Only create causal mask buffer if we're not using Flash Attention
        # (Flash handles causal masking internally)
        if is_causal and not use_flash_attention:
            # Buffer: fixed causal mask that prevents attention to future tokens
            # Not a learned parameter - just a precomputed mask for efficiency
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_seq_length, max_seq_length)).view(
                    1, 1, max_seq_length, max_seq_length
                ),
            )
        else:
            self.causal_mask = None

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Dropout modules - always create them, PyTorch handles p=0 or p=None efficiently
        self.attn_dropout = nn.Dropout(dropout if dropout is not None else 0.0)
        self.resid_dropout = nn.Dropout(dropout if dropout is not None else 0.0)

    def _apply_output_projection(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Apply output projection."""
        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self.output_projection(x)
        return output

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply standard multi-head attention.

        The MHA process:
        1. Combined projection - single matrix multiply produces Q, K, V
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
        q, k, v = self._compute_qkv_projections(x, self.qkv_projection, self.hidden_dim)

        q_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        k_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        v_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        q_heads = self._reshape_to_multihead(q, batch_size, seq_len, self.num_heads, self.head_dim)
        k_heads = self._reshape_to_multihead(k, batch_size, seq_len, self.num_heads, self.head_dim)
        v_heads = self._reshape_to_multihead(v, batch_size, seq_len, self.num_heads, self.head_dim)

        attn_output: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        if self.use_flash_attention:
            attn_output = self._apply_flash_attention(
                q_heads,
                k_heads,
                v_heads,
                attention_mask=attention_mask,
                dropout_p=self.attn_dropout.p,
                is_causal=self.is_causal,
                enable_gqa=False,
            )
        else:
            attn_output = self._compute_manual_attention(
                q_heads,
                k_heads,
                v_heads,
                head_dim=self.head_dim,
                is_causal=self.is_causal,
                attention_scale=self.attention_scale,
                attention_mask=attention_mask,
                causal_mask=self.causal_mask,
                attn_dropout=self.attn_dropout,
            )

        merged: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        merged = self._merge_heads(attn_output, self.hidden_dim)

        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self._apply_output_projection(merged)
        output = self.resid_dropout(output)

        return output
