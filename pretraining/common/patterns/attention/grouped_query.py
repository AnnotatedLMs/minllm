# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from pretraining.common.patterns.attention import base
from pretraining.common.patterns.cache import kv_cache
from pretraining.common.patterns.position import rope


class GroupedQueryAttention(base.Attention):
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
        rope_module: rope.RoPE,  # Preferred for Llama-style models
        dropout: float = 0.0,
        bias: bool = False,
        is_causal: bool = True,
        use_flash_attention: bool = True,
    ):
        # Initialize base with query heads count
        super().__init__(hidden_dim, num_heads, dropout, is_causal, use_flash_attention)

        self.num_kv_heads = num_kv_heads
        self.rope_module = rope_module
        assert num_heads % num_kv_heads == 0

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

        # Optional KVCache for inference optimization
        # This is initialized as None but can be set by the parent LLM
        # via install_kv_cache() which directly assigns to transformer.attention.cache
        self.cache: typing.Optional[kv_cache.KVCache] = None
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def _compute_projections(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads*head_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads*head_dim"],
    ]:
        """
        Compute Q, K, V projections.

        GQA specific: K and V have fewer heads than Q.
        """
        # Query projection - full dimension
        q: jaxtyping.Float[torch.Tensor, "batch seq d_model"] = self.q_proj(x)
        # Key/Value projections - reduced dimension
        k: jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads*head_dim"] = self.k_proj(x)
        v: jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads*head_dim"] = self.v_proj(x)

        return q, k, v

    def _reshape_queries_to_multihead(
        self,
        q: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        batch_size: int,
        seq_len: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]:
        """Reshape queries to multi-head format."""
        q_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        q_heads = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q_heads = q_heads.transpose(1, 2)
        return q_heads

    def _reshape_kv_to_multihead(
        self,
        tensor: jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads*head_dim"],
        batch_size: int,
        seq_len: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"]:
        """Reshape key/value tensors to multi-head format with fewer heads."""
        kv_heads: jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"]
        kv_heads = tensor.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        kv_heads = kv_heads.transpose(1, 2)
        return kv_heads

    def _repeat_kv(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads head_dim"],
        n_rep: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq n_heads head_dim"]:
        """
        Repeat key/value heads to match number of query heads.

        Used in Grouped Query Attention (GQA) where n_kv_heads < n_heads.
        For example, if n_heads=32 and n_kv_heads=8, each kv head is
        repeated 4 times.

        Args:
            x: Key or value tensor with n_kv_heads
            n_rep: Repetition factor (n_heads // n_kv_heads)

        Returns:
            Tensor with repeated heads to match query head count
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
        Apply rotary positional embeddings using StandardRoPE.

        GroupedQueryAttention is architecturally tied to StandardRoPE
        which applies RoPE to full semantic vectors.

        Args:
            q: Query tensor
            k: Key tensor
            position_offset: Starting position for RoPE (for KV caching)
        """
        # StandardRoPE expects [batch, seq, heads, head_dim]
        q_for_rope = q.transpose(1, 2)  # [batch, seq, n_heads, head_dim]
        k_for_rope = k.transpose(1, 2)  # [batch, seq, n_kv_heads, head_dim]

        # Apply StandardRoPE with position offset
        q_rot = self.rope_module(q_for_rope, position_offset)
        k_rot = self.rope_module(k_for_rope, position_offset)

        # Transpose back to [batch, n_heads, seq, head_dim]
        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)

        return q_rot, k_rot

    def _merge_heads(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Merge attention heads back to single tensor."""
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Transpose heads and sequence dimensions
        x_transposed: jaxtyping.Float[torch.Tensor, "batch seq n_heads head_dim"]
        x_transposed = x.transpose(1, 2).contiguous()

        # Reshape to combine heads
        x_merged: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        x_merged = x_transposed.view(batch_size, seq_len, self.hidden_dim)

        return x_merged

    def _apply_output_projection(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Apply output projection."""
        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self.o_proj(x)
        return output

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
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

        q: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        k: jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads*head_dim"]
        v: jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads*head_dim"]
        q, k, v = self._compute_projections(x)

        q_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        q_heads = self._reshape_queries_to_multihead(q, batch_size, seq_len)

        k_heads: jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"]
        k_heads = self._reshape_kv_to_multihead(k, batch_size, seq_len)

        v_heads: jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"]
        v_heads = self._reshape_kv_to_multihead(v, batch_size, seq_len)

        q_heads, k_heads = self._apply_rotary_embeddings(q_heads, k_heads, position_offset)

        # Handle KV cache if installed
        if self.cache is not None:
            # KV cache expects [batch, seq, n_kv_heads, head_dim]
            # but we have [batch, n_kv_heads, seq, head_dim]
            k_for_cache = k_heads.transpose(1, 2)
            v_for_cache = v_heads.transpose(1, 2)

            # Update cache and get back accumulated keys/values
            k_cached, v_cached = self.cache.update(position_offset, k_for_cache, v_for_cache)

            # Transpose back to [batch, n_kv_heads, seq, head_dim]
            k_heads = k_cached.transpose(1, 2)
            v_heads = v_cached.transpose(1, 2)

        # Repeat KV heads to match Q heads
        k_repeated: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        v_repeated: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        k_repeated = self._repeat_kv_heads(k_heads)
        v_repeated = self._repeat_kv_heads(v_heads)

        attn_output: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        # Flash Attention doesn't support different sequence lengths for Q and K/V
        # When using KV cache, Q has seq_len=1 but K/V have seq_len=position_offset+1
        use_flash = self.use_flash_attention and (
            self.cache is None or q_heads.size(2) == k_repeated.size(2)
        )

        if use_flash:
            attn_output = self._apply_flash_attention(
                q_heads, k_repeated, v_repeated, attention_mask
            )
        else:
            attn_output = self._compute_manual_attention(
                q_heads, k_repeated, v_repeated, attention_mask
            )

        merged: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        merged = self._merge_heads(attn_output)

        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self._apply_output_projection(merged)

        return output
