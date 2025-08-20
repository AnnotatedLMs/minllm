# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn


class FusedQKVProjectionMixin:
    """
    Mixin that provides fused QKV projection for attention mechanisms.

    Variation: Fused/combined QKV projection (standard in MHA)
    Computation: Single linear layer produces all three projections
    Effect: More efficient than separate projections, better memory locality

    Used by: Multi-head attention (GPT-2, BERT, etc.)
    """

    def _compute_qkv_projections(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        qkv_projection: nn.Linear,
        hidden_dim: int,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ]:
        """
        Compute Q, K, V projections using fused projection matrix.

        Single linear layer produces all three projections at once,
        then splits them into separate tensors.

        Args:
            x: Input tensor
            qkv_projection: Fused QKV projection layer (projects to 3*hidden_dim)
            hidden_dim: Hidden dimension for splitting QKV
        """
        # Combined projection to 3x hidden dimension
        qkv: jaxtyping.Float[torch.Tensor, "batch seq_len 3*hidden_dim"] = qkv_projection(x)

        # Split into Q, K, V
        q: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        k: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        v: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        q, k, v = qkv.split(hidden_dim, dim=2)

        return q, k, v


class GroupedQKVProjectionMixin:
    """
    Mixin that provides grouped QKV projection for memory-efficient attention.

    Variation: Separate projections with different dimensions for K/V
    Computation: Q gets full dimension, K/V get reduced dimension
    Effect: Reduces KV cache memory by num_heads/num_kv_heads factor

    Used by: Grouped Query Attention (Llama 3, etc.)
    """

    def _compute_grouped_qkv_projections(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq_len n_kv_heads*head_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq_len n_kv_heads*head_dim"],
    ]:
        """
        Compute Q, K, V projections with grouped dimensions.

        Query gets full dimension while Key/Value get reduced dimension
        to enable memory-efficient grouped query attention.

        Args:
            x: Input tensor
            q_proj: Query projection (to hidden_dim)
            k_proj: Key projection (to n_kv_heads * head_dim)
            v_proj: Value projection (to n_kv_heads * head_dim)
        """
        # Query projection - full dimension
        q: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"] = q_proj(x)

        # Key/Value projections - reduced dimension
        k: jaxtyping.Float[torch.Tensor, "batch seq_len n_kv_heads*head_dim"] = k_proj(x)
        v: jaxtyping.Float[torch.Tensor, "batch seq_len n_kv_heads*head_dim"] = v_proj(x)

        return q, k, v


class MLAProjectionMixin:
    """
    Mixin for MLA projection from compressed to attention dimensions.

    Significance:
    Takes tiny compressed representations and expands them back to full attention size.
    Keeps content (what the token says) separate from position (where the token is).

    Init:
    The projection layers are defined in MultiHeadLatentAttention as:
        self.key_up = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.value_up = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.query_up = nn.Linear(query_compression_dim, num_heads * head_dim)
        self.key_rope = nn.Linear(hidden_dim, num_heads * rope_dim)
        self.query_rope = nn.Linear(query_compression_dim, num_heads * rope_dim)

    Step-by-step control flow:
    1. Take compressed KV and expand to full-size keys and values
    2. Take original input and project to position features for keys
    3. Take compressed Q and expand to full-size queries and position features
    4. Return all components separately for later combination

    Learning process:
    - Up-projections learn to reconstruct full dimensions from compressed versions
    - Content projections learn semantic patterns that matter for attention
    - Position projections learn which position relationships matter

    Used by: DeepSeek-V3's Multi-head Latent Attention
    """

    def _project_compressed_kv(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],  # Original input for RoPE
        kv_compressed: jaxtyping.Float[torch.Tensor, "batch seq_len kv_compress_dim"],
        key_up_proj: nn.Linear,
        value_up_proj: nn.Linear,
        key_rope_proj: nn.Linear,
        num_heads: int,
        head_dim: int,
        rope_dim: int,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq_len num_heads*head_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq_len num_heads*head_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq_len num_heads*rope_dim"],
    ]:
        """Project compressed KV to keys, values, and RoPE components."""
        # Project to content keys
        keys_content: jaxtyping.Float[torch.Tensor, "batch seq_len num_heads*head_dim"]
        keys_content = key_up_proj(kv_compressed)

        # Project to values
        values: jaxtyping.Float[torch.Tensor, "batch seq_len num_heads*head_dim"]
        values = value_up_proj(kv_compressed)

        # Project to RoPE keys
        keys_rope: jaxtyping.Float[torch.Tensor, "batch seq_len num_heads*rope_dim"]
        keys_rope = key_rope_proj(x)

        return keys_content, values, keys_rope

    def _project_compressed_queries(
        self,
        q_compressed: jaxtyping.Float[torch.Tensor, "batch seq_len q_compress_dim"],
        query_up_proj: nn.Linear,
        query_rope_proj: nn.Linear,
        num_heads: int,
        head_dim: int,
        rope_dim: int,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq_len num_heads*head_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq_len num_heads*rope_dim"],
    ]:
        """Project compressed queries to content and RoPE components."""
        # Project to content queries
        queries_content: jaxtyping.Float[torch.Tensor, "batch seq_len num_heads*head_dim"]
        queries_content = query_up_proj(q_compressed)

        # Project to RoPE queries
        queries_rope: jaxtyping.Float[torch.Tensor, "batch seq_len num_heads*rope_dim"]
        queries_rope = query_rope_proj(q_compressed)

        return queries_content, queries_rope
