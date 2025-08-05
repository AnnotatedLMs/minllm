# Standard Library
import math
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from pretraining.common.patterns.attention import core
from pretraining.common.patterns.position import rope_partial


class MultiHeadLatentAttention(core.Attention):
    """
    Multi-head Latent Attention (MLA) - Compression-based attention for large models.

    Used by: DeepSeek-V3

    Variation: Compresses Q, K, V through a bottleneck before expanding to attention dimensions
    Computation: Input → small latent → separate projections for K, V, Q, and RoPE
    Effect: Model can scale to very large sizes with manageable activation memory

    Variation: Decouples content (key/value) from position (RoPE) dimensions
    Computation: RoPE is computed on a subset of dimensions, then concatenated
    Effect: Model learns to separate semantic content from positional relationships

    See forward() for the compression → projection → attention flow.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        kv_compression_dim: int,
        query_compression_dim: int,
        rope_module: rope_partial.PartialRoPE,  # Required for DeepSeek-style models
        rope_dim: int = 64,
        dropout: float = 0.0,
        is_causal: bool = True,
        use_flash_attention: bool = True,
    ):
        # Note: MLA uses explicit head_dim rather than deriving from hidden_dim
        super().__init__(hidden_dim, num_heads, dropout, is_causal, use_flash_attention)

        # Override head_dim
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.rope_module = rope_module

        # Compression layers
        self.kv_down = nn.Linear(hidden_dim, kv_compression_dim)
        self.query_down = nn.Linear(hidden_dim, query_compression_dim)

        # Separate up-projections for keys, values, and queries
        self.key_up = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.value_up = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.query_up = nn.Linear(query_compression_dim, num_heads * head_dim)

        # Separate RoPE projections
        self.key_rope = nn.Linear(kv_compression_dim, num_heads * rope_dim)
        self.query_rope = nn.Linear(query_compression_dim, num_heads * rope_dim)

        self.output_proj = nn.Linear(num_heads * head_dim, hidden_dim)

    def _compute_attention_scores(
        self,
        q: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"],
        k: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """
        Compute raw attention scores: Q @ K^T / scale.

        Override: MLA scales by total dimension (content + RoPE).
        """
        # Compute dot product
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Apply MLA-specific scaling
        scale: float = math.sqrt(self.head_dim + self.rope_dim)
        scaled_scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        scaled_scores = scores / scale

        return scaled_scores

    def _compress_inputs(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq kv_compress_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq q_compress_dim"],
    ]:
        """
        Compress inputs to latent dimensions.

        MLA specific: Compression before projection saves memory.
        """
        kv_compressed: jaxtyping.Float[torch.Tensor, "batch seq kv_compress_dim"]
        kv_compressed = self.kv_down(x)

        query_compressed: jaxtyping.Float[torch.Tensor, "batch seq q_compress_dim"]
        query_compressed = self.query_down(x)

        return kv_compressed, query_compressed

    def _project_compressed_kv(
        self,
        kv_compressed: jaxtyping.Float[torch.Tensor, "batch seq kv_compress_dim"],
        batch_size: int,
        seq_len: int,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"],
        jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"],
        jaxtyping.Float[torch.Tensor, "batch n_heads seq rope_dim"],
    ]:
        """Project compressed KV to keys, values, and RoPE components."""
        # Project to content keys
        keys_content: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads*head_dim"]
        keys_content = self.key_up(kv_compressed)

        keys_content_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        keys_content_heads = keys_content.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys_content_heads = keys_content_heads.transpose(1, 2)

        # Project to values
        values: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads*head_dim"]
        values = self.value_up(kv_compressed)

        values_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        values_heads = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values_heads = values_heads.transpose(1, 2)

        # Project to RoPE keys
        keys_rope: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads*rope_dim"]
        keys_rope = self.key_rope(kv_compressed)

        keys_rope_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq rope_dim"]
        keys_rope_heads = keys_rope.view(batch_size, seq_len, self.num_heads, self.rope_dim)
        keys_rope_heads = keys_rope_heads.transpose(1, 2)

        return keys_content_heads, values_heads, keys_rope_heads

    def _project_compressed_queries(
        self,
        query_compressed: jaxtyping.Float[torch.Tensor, "batch seq q_compress_dim"],
        batch_size: int,
        seq_len: int,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"],
        jaxtyping.Float[torch.Tensor, "batch n_heads seq rope_dim"],
    ]:
        """Project compressed queries to content and RoPE components."""
        # Project to content queries
        queries_content: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads*head_dim"]
        queries_content = self.query_up(query_compressed)

        queries_content_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        queries_content_heads = queries_content.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        queries_content_heads = queries_content_heads.transpose(1, 2)

        # Project to RoPE queries
        queries_rope: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads*rope_dim"]
        queries_rope = self.query_rope(query_compressed)

        queries_rope_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq rope_dim"]
        queries_rope_heads = queries_rope.view(batch_size, seq_len, self.num_heads, self.rope_dim)
        queries_rope_heads = queries_rope_heads.transpose(1, 2)

        return queries_content_heads, queries_rope_heads

    def _apply_rope_to_subset(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch n_heads seq rope_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq rope_dim"]:
        """
        Apply RoPE to a subset of dimensions using PartialRoPE.

        MLA specific: Only rope_dim dimensions get positional encoding.
        MultiHeadLatentAttention is architecturally tied to PartialRoPE
        which applies RoPE to position-only vectors.
        """
        # PartialRoPE expects [batch, heads, seq, rope_dim] - already in correct format!
        x_rot = self.rope_module(x)

        return x_rot

    def _merge_heads(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len n_heads*head_dim"]:
        """
        Merge attention heads back to single tensor.

        Override: MLA has explicit head_dim, so output is n_heads*head_dim
        which may differ from hidden_dim.
        """
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Transpose heads and sequence dimensions
        x_transposed: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads head_dim"]
        x_transposed = x.transpose(1, 2).contiguous()

        # Reshape to combine heads
        x_merged: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads*head_dim"]
        x_merged = x_transposed.view(batch_size, seq_len, self.num_heads * self.head_dim)

        return x_merged

    def _apply_output_projection(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads*head_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Apply output projection to map back to model dimension."""
        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self.output_proj(x)
        return output

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply Multi-head Latent Attention.

        The MLA process:
        1. Compress inputs - reduce to smaller latent dimensions
        2. Separate projections - compute content (K, V, Q) and position (K_rope, Q_rope) separately
        3. Apply RoPE - rotate only the position-specific dimensions
        4. Concatenate - combine content and rotated position features
        5. Apply attention - use concatenated Q/K with values
        6. Output projection - map back to model dimension

        Note: No KV caching due to the compression step.
        """
        # Note: MLA does not support KV caching due to compression
        # DeepSeek uses MLA without position_offset since PartialRoPE
        # computes positions dynamically

        batch_size, seq_len, hidden_dim = x.shape

        kv_compressed: jaxtyping.Float[torch.Tensor, "batch seq kv_compress_dim"]
        query_compressed: jaxtyping.Float[torch.Tensor, "batch seq q_compress_dim"]
        kv_compressed, query_compressed = self._compress_inputs(x)

        keys_content: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        values: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        keys_rope: jaxtyping.Float[torch.Tensor, "batch n_heads seq rope_dim"]
        keys_content, values, keys_rope = self._project_compressed_kv(
            kv_compressed, batch_size, seq_len
        )

        queries_content: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        queries_rope: jaxtyping.Float[torch.Tensor, "batch n_heads seq rope_dim"]
        queries_content, queries_rope = self._project_compressed_queries(
            query_compressed, batch_size, seq_len
        )

        # Apply RoPE to the rope dimensions
        keys_rope = self._apply_rope_to_subset(keys_rope)
        queries_rope = self._apply_rope_to_subset(queries_rope)

        # Concatenate content and position features
        queries: jaxtyping.Float[torch.Tensor, "batch n_heads seq total_dim"]
        queries = torch.cat([queries_content, queries_rope], dim=-1)

        keys: jaxtyping.Float[torch.Tensor, "batch n_heads seq total_dim"]
        keys = torch.cat([keys_content, keys_rope], dim=-1)

        # Note: We use the concatenated Q/K but only values for output
        attn_output: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        # MLA doesn't support Flash Attention due to concatenated dimensions
        # Always use manual attention regardless of config
        attn_output = self._compute_manual_attention(queries, keys, values, attention_mask)

        merged: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads*head_dim"]
        merged = self._merge_heads(attn_output)

        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = self._apply_output_projection(merged)

        return output
