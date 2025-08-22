# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn


class FusedQKVProjectionMixin:
    """
    Mixin for fused Query, Key, Value projections in multi-head attention.

    Scholarship:
    - Vaswani et al., Attention Is All You Need, 2017. https://arxiv.org/pdf/1706.03762

    Significance:
    Enables parallel attention heads by projecting inputs into multiple representation subspaces.
    Single fused projection is more efficient than three separate projections.
    Each head learns to attend to different types of relationships between tokens.

    Init:
    This mixin has no initialization. It works with components defined in the attention module:
        self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * head_dim)  # Fused QKV projection

    Step-by-step control flow (_compute_qkv_projections):
    1. Apply single linear transformation producing 3x the hidden dimension
    2. Split the output into three equal parts for Q, K, V
    3. Return separate Q, K, V tensors for attention computation

    Learning process:
    - Fused QKV projection (self.qkv_proj: nn.Linear):
      - Learns three distinct transformations within a single weight matrix
      - Query portion learns to extract "what am I looking for" features
      - Key portion learns to extract "what do I contain" features
      - Value portion learns to extract "what information to pass forward" features

      - When predictions are wrong: loss increases, producing gradients
      - Gradients flow back through attention scores to all three projections simultaneously
      - Q weights adjust to better identify what current token needs from context
      - K weights adjust to better advertise what each token offers
      - V weights adjust to better package useful information for next layers
      - Result: Single matrix learns three complementary transformations that work together

    - Efficiency benefit:
      - Single matrix multiplication is faster than three separate ones
      - Better memory locality and cache utilization
      - Gradients for Q, K, V are computed together, improving optimization
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

    Scholarship:
    Deepseek-V2, 2024, https://arxiv.org/pdf/2405.04434, 2.1.2
    Deepseek-V3, 2025, https://arxiv.org/pdf/2412.19437, 2.1.1

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
    - Key up-projection:
      - Learns to expand compressed KV into keys that differentiate tokens for attention
      - When attention patterns are wrong: gradients signal that expanded keys don't distinguish relevant from irrelevant tokens
      - Weight matrix adjusts to produce keys that make important tokens stand out in dot products
      - Result: transformation learns to reconstruct discriminative features from compressed representation

    - Value up-projection:
      - Learns to expand compressed KV into values containing useful information
      - When predictions fail: gradients signal that expanded values lack necessary content
      - Weight matrix adjusts to reconstruct information dimensions that improve next-token prediction
      - Result: transformation learns to unpack predictive features from compressed representation

    - Query up-projection:
      - Learns to expand compressed queries to match with relevant keys
      - When wrong tokens get attention: gradients signal that expanded queries align with unhelpful keys
      - Weight matrix adjusts to produce queries that have higher dot products with useful token keys
      - Result: transformation learns to reconstruct query features that find relevant context

    - Position projections (RoPE):
      - Key RoPE: Learns which positional relationships help queries find tokens
        - Takes uncompressed input to preserve full positional information
        - Weight matrix learns to extract position features that matter for token relevance
      - Query RoPE: Learns positional features for finding tokens at relevant distances
        - Works from compressed queries but focuses on position-specific dimensions
        - Weight matrix learns to produce position encodings that identify useful relative positions

    - All projections jointly optimize through attention scores to balance content and position matching
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
