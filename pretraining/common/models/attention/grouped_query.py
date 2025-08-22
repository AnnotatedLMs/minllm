# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models.attention import attention_mixins
from pretraining.common.models.attention import cache_mixins
from pretraining.common.models.attention import projection_mixins
from pretraining.common.models.attention import reshape_mixins
from pretraining.common.models.position import rope


class GroupedQueryAttention(
    nn.Module,
    projection_mixins.GroupedQKVProjectionMixin,
    reshape_mixins.MultiHeadReshapeMixin,
    attention_mixins.ManualSDPAMixin,
    attention_mixins.FlashAttentionMixin,
    cache_mixins.CachedAttentionMixin,
):
    """
    Grouped Query Attention (GQA) - Memory-efficient attention through KV sharing.

    Used by: Llama 3 models

    Variation: Groups of query heads share the same key-value pairs
    Computation: K and V matrices are smaller (fewer heads), then repeated to match Q heads
    Effect: Maintains model quality while reducing memory footprint during inference

    Variation: Uses RoPE (rotary position embeddings) instead of learned position embeddings
    Computation: Position information is applied by rotating Q and K vectors
    Effect: Model naturally understands relative positions and can generalize to any sequence length

    See forward() for the specific flow including KV repetition and caching.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_module: rope.PrecomputedRoPE,
        dropout: typing.Optional[float] = None,
        bias: bool = False,
        is_causal: bool = True,
        use_flash_attention: bool = True,
        attention_scale: typing.Optional[float] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.rope_module = rope_module
        self.is_causal = is_causal
        self.use_flash_attention = use_flash_attention
        self.attention_scale = attention_scale

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        # Validate RoPE dimension matches head dimension
        if rope_module.dim != self.head_dim:
            raise ValueError(
                f"RoPE dimension ({rope_module.dim}) must match head dimension ({self.head_dim}). "
                f"Got hidden_dim={hidden_dim}, num_heads={num_heads}, head_dim={self.head_dim}"
            )

        self.n_rep = num_heads // num_kv_heads  # Repetition factor

        # GQA projections - fewer KV heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Dropout modules
        self.attn_dropout = nn.Dropout(dropout if dropout is not None else 0.0)
        self.resid_dropout = nn.Dropout(dropout if dropout is not None else 0.0)

    def _repeat_kv(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len n_kv_heads head_dim"],
        n_rep: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len n_heads head_dim"]:
        """
        Repeat key/value heads to match number of query heads.

        Used in Grouped Query Attention (GQA) where n_kv_heads < n_heads.
        For example, if n_heads=32 and n_kv_heads=8, each kv head is
        repeated 4 times.
        """
        if n_rep == 1:
            return x

        batch_size, seq_len, n_kv_heads, head_dim = x.shape

        # Expand and reshape to repeat each head n_rep times
        # [batch_size, seq_len, n_kv_heads, head_dim]
        # -> [batch_size, seq_len, n_kv_heads, 1, head_dim]
        # -> [batch_size, seq_len, n_kv_heads, n_rep, head_dim]
        # -> [batch_size, seq_len, n_kv_heads * n_rep, head_dim]
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

    def _repeat_kv_heads(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]:
        """
        Repeat KV heads to match number of Q heads.

        GQA specific: Each KV head is shared by n_rep query heads.
        """
        if self.n_rep == 1:
            return x

        # Transpose to match expected input format for repeat_kv
        # [batch, n_kv_heads, seq, head_dim] -> [batch, seq, n_kv_heads, head_dim]
        x_transposed = x.transpose(1, 2)

        x_repeated = self._repeat_kv(x_transposed, self.n_rep)

        # Transpose back
        # [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
        x_final = x_repeated.transpose(1, 2)

        return x_final

    def _apply_rotary_embeddings(
        self,
        q: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"],
        k: jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"],
        position_offset: int = 0,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"],
        jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"],
    ]:
        """
        Apply rotary positional embeddings using RoPE.
        """
        # RoPE expects [batch, seq, heads, head_dim]
        q_for_rope = q.transpose(1, 2)  # [batch, seq, n_heads, head_dim]
        k_for_rope = k.transpose(1, 2)  # [batch, seq, n_kv_heads, head_dim]

        # Apply RoPE with position offset
        q_rot: torch.Tensor = self.rope_module(q_for_rope, position_offset)
        k_rot: torch.Tensor = self.rope_module(k_for_rope, position_offset)

        # Transpose back to [batch, n_heads, seq, head_dim]
        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)

        return q_rot, k_rot

    def _apply_output_projection(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Apply output projection."""
        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self.o_proj(x)
        return output

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply grouped query attention.

        The GQA process:
        1. Separate projections - Q gets full heads, K/V get fewer heads to save memory
        2. Apply RoPE - rotate Q and K vectors based on their absolute positions
        3. Handle KV cache - store/retrieve past keys and values for efficient generation
        4. Repeat K/V heads - each KV pair serves multiple query heads
        5. Apply attention - standard attention but with repeated K/V
        6. Output projection - linear transform back to model dimension
        """
        batch_size, seq_len, hidden_dim = x.shape

        q: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        k: jaxtyping.Float[torch.Tensor, "batch seq_len n_kv_heads*head_dim"]
        v: jaxtyping.Float[torch.Tensor, "batch seq_len n_kv_heads*head_dim"]
        q, k, v = self._compute_grouped_qkv_projections(x, self.q_proj, self.k_proj, self.v_proj)

        # Reshape Q using the standard mixin (full heads)
        q_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        q_heads = self._reshape_to_multihead(q, batch_size, seq_len, self.num_heads, self.head_dim)

        # Reshape K/V using the standard mixin (fewer heads)
        k_heads: jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"]
        k_heads = self._reshape_to_multihead(
            k, batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        v_heads: jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"]
        v_heads = self._reshape_to_multihead(
            v, batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        q_heads, k_heads = self._apply_rotary_embeddings(q_heads, k_heads, position_offset)

        # Handle KV cache if installed
        if self.has_cache:
            k_heads, v_heads = self.update_kv_cache_with_transpose(
                k_heads, v_heads, position_offset
            )

        # Repeat KV heads to match Q heads
        k_repeated: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        v_repeated: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        k_repeated = self._repeat_kv_heads(k_heads)
        v_repeated = self._repeat_kv_heads(v_heads)

        attn_output: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        # Flash Attention doesn't support different sequence lengths for Q and K/V
        # When using KV cache, Q has seq_len=1 but K/V have seq_len=position_offset+1
        use_flash = self.use_flash_attention and (
            not self.has_cache or q_heads.size(2) == k_repeated.size(2)
        )

        if use_flash:
            attn_output = self._apply_flash_attention(
                q_heads,
                k_repeated,
                v_repeated,
                attention_mask=attention_mask,
                dropout_p=self.attn_dropout.p,
                is_causal=self.is_causal,
                enable_gqa=True,
            )
        else:
            attn_output = self._compute_manual_attention(
                q_heads,
                k_repeated,
                v_repeated,
                head_dim=self.head_dim,
                is_causal=self.is_causal,
                attention_scale=self.attention_scale,
                attention_mask=attention_mask,
                causal_mask=None,  # GQA uses dynamic mask creation
                attn_dropout=self.attn_dropout,
            )

        merged: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        merged = self._merge_heads(attn_output, self.hidden_dim)

        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self._apply_output_projection(merged)
        output = self.resid_dropout(output)

        return output
