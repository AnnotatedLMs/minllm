# Standard Library
import math
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import attention as torch_attention

# Project
# Local
from pretraining.common.base.models import attention
from pretraining.common.patterns.components import cache
from pretraining.common.patterns.components import position

# TODO: Flash attn implementation for multi-head latent attention?
# TODO: Did we do caching right for MultiHeadLatentAttention?

# TODO: # Optional KVCache for inference optimization self.cache: typing.Optional['cache.KVCache'] = None
# TODO: Llama builds the cache in the constructor, but unclear how this gets passed to gqa


class Attention(attention.BaseAttention):
    """
    Class for attention patterns with common implementations.

    This class provides standard implementations of attention operations
    that are shared across most attention mechanisms.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_causal: bool = True,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.is_causal = is_causal
        self.use_flash_attention = use_flash_attention

        assert hidden_dim % num_heads == 0

        self.attn_dropout = nn.Dropout(dropout)

    def _compute_attention_scores(
        self,
        q: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"],
        k: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """
        Compute raw attention scores: Q @ K^T / scale.

        Standard scaled dot-product attention.
        """
        # Compute dot product
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Apply scaling
        scale: float = math.sqrt(self.head_dim)
        scaled_scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        scaled_scores = scores / scale

        return scaled_scores

    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> jaxtyping.Bool[torch.Tensor, "seq seq"]:
        """
        Create lower triangular causal mask.

        Standard implementation for autoregressive models.
        """
        mask: jaxtyping.Bool[torch.Tensor, "seq seq"]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask

    def _apply_causal_mask(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
        seq_len: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """Apply causal mask to attention scores."""
        if not self.is_causal:
            return scores

        # Create causal mask
        causal_mask: jaxtyping.Bool[torch.Tensor, "seq seq"]
        causal_mask = self._create_causal_mask(seq_len, scores.device)

        # Apply mask (set future positions to -inf)
        masked_scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        masked_scores = scores.masked_fill(~causal_mask, float("-inf"))

        return masked_scores

    def _apply_attention_mask(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
        attention_mask: typing.Optional[torch.Tensor],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """Apply optional attention mask (e.g., for padding)."""
        if attention_mask is None:
            return scores

        # Apply padding mask
        masked_scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        masked_scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        return masked_scores

    def _compute_attention_probs(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """Convert attention scores to probabilities."""
        attn_probs: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        attn_probs = F.softmax(scores, dim=-1)
        return attn_probs

    def _apply_attention_dropout(
        self,
        attn_probs: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """Apply dropout to attention probabilities."""
        dropped_probs: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        dropped_probs = self.attn_dropout(attn_probs)
        return dropped_probs

    def _compute_weighted_values(
        self,
        attn_probs: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
        v: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]:
        """Compute weighted sum of values using attention probabilities."""
        output: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]
        output = torch.matmul(attn_probs, v)
        return output

    def _apply_flash_attention(
        self,
        q: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"],
        k: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
        v: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]:
        """
        Apply Flash Attention using PyTorch's scaled_dot_product_attention.

        Uses context manager to explicitly request Flash Attention backend.
        Will fall back to other backends if Flash Attention is not available.
        """
        if not self.use_flash_attention:
            raise RuntimeError("Flash Attention not enabled for this module")

        output: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]

        # Try to use Flash Attention with context manager
        with torch_attention.sdpa_kernel(backends=[torch_attention.SDPBackend.FLASH_ATTENTION]):
            # For GroupedQueryAttention, enable GQA optimization
            enable_gqa = hasattr(self, "num_kv_heads") and self.num_kv_heads != self.num_heads

            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal
                and attention_mask is None,  # Only use is_causal if no custom mask
                enable_gqa=enable_gqa,  # Enable GQA optimization for grouped query attention
            )

        return output

    def _compute_manual_attention(
        self,
        q: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"],
        k: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
        v: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]:
        """
        Compute attention manually (non-Flash path).

        The standard process:
        1. Compute attention scores
        2. Apply causal mask
        3. Apply attention mask (if provided)
        4. Convert to probabilities
        5. Apply dropout
        6. Apply attention to values
        """
        seq_len = q.size(2)

        # Step 1: Compute attention scores
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        scores = self._compute_attention_scores(q, k)

        # Step 2: Apply causal mask
        scores = self._apply_causal_mask(scores, seq_len)

        # Step 3: Apply attention mask if provided
        scores = self._apply_attention_mask(scores, attention_mask)

        # Step 4: Convert to probabilities
        attn_probs: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        attn_probs = self._compute_attention_probs(scores)

        # Step 5: Apply dropout
        attn_probs = self._apply_attention_dropout(attn_probs)

        # Step 6: Compute weighted values
        output: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]
        output = self._compute_weighted_values(attn_probs, v)

        return output


class MultiHeadAttention(Attention):
    """
    Multi-Head Attention pattern.

    Used by: GPT-2 (with combined QKV projection).
    Pattern: Combined QKV projection → Split → Multi-head attention → Output projection

    This is the classic attention mechanism where all heads have their own
    key and value projections. GPT-2 uses learned position embeddings added
    to the input, so no RoPE is needed here.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
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

        self.resid_dropout = nn.Dropout(dropout)

        # Only create bias buffer if we're not using Flash Attention
        # (Flash handles causal masking internally)
        if not self.use_flash_attention:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(max_seq_length, max_seq_length)).view(
                    1, 1, max_seq_length, max_seq_length
                ),
            )

    def _compute_qkv_projections(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ]:
        """
        Compute Q, K, V projections using combined projection matrix.

        GPT-2 specific: Single linear layer produces all three projections.
        """
        # Combined projection to 3x hidden dimension
        qkv: jaxtyping.Float[torch.Tensor, "batch seq 3*d_model"] = self.c_attn(x)

        # Split into Q, K, V
        q: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        k: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        v: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        q, k, v = qkv.split(self.hidden_dim, dim=2)

        return q, k, v

    def _reshape_to_multihead(
        self,
        tensor: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        batch_size: int,
        seq_len: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]:
        """Reshape tensor for multi-head attention computation."""

        # First reshape to separate head dimension
        reshaped: jaxtyping.Float[torch.Tensor, "batch seq n_heads head_dim"]
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
        """Apply output projection and dropout."""
        # Linear projection
        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self.c_proj(x)

        # Residual dropout
        output = self.resid_dropout(output)

        return output

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply standard multi-head attention.

        Note: GPT-2 uses learned position embeddings added to the input
        before the transformer blocks, so no RoPE is needed.

        The process:
        1. Extract dimensions from input tensor
        2. Compute Q, K, V projections using combined projection
        3. Reshape projections for multi-head format
        4. Apply attention (Flash or manual)
        5. Merge heads and apply output projection
        """

        batch_size, seq_len, hidden_dim = x.shape

        q: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        k: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        v: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
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

        merged: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        merged = self._merge_heads(attn_output)

        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self._apply_output_projection(merged)

        return output


class GroupedQueryAttention(Attention):
    """
    Grouped Query Attention (GQA) pattern.

    Used by: Llama 3 70B (8 KV heads for 64 Q heads).
    Pattern: Q projection (all heads) → K/V projection (fewer heads) → Apply RoPE → Repeat K/V → Attention

    GQA reduces memory usage by sharing key and value heads across multiple
    query heads. For example, with 32 query heads and 8 KV heads, each KV
    head is shared by 4 query heads.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_module: "position.PrecomputedRoPE",  # Required for Llama-style models
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
        self.cache: typing.Optional[cache.KVCache] = None
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
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]:
        """Reshape queries to multi-head format."""
        batch_size, seq_len, d_model = q.shape

        q_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        q_heads = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q_heads = q_heads.transpose(1, 2)
        return q_heads

    def _reshape_kv_to_multihead(
        self,
        tensor: jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads*head_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"]:
        """Reshape key/value tensors to multi-head format with fewer heads."""
        batch_size, seq_len, kv_dim = tensor.shape

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

        bs, slen, n_kv_heads, head_dim = x.shape

        # Expand and reshape to repeat each head n_rep times
        # [bs, slen, n_kv_heads, head_dim]
        # -> [bs, slen, n_kv_heads, 1, head_dim]
        # -> [bs, slen, n_kv_heads, n_rep, head_dim]
        # -> [bs, slen, n_kv_heads * n_rep, head_dim]
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
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

        The process:
        1. Project Q with all heads, K/V with fewer heads
        2. Reshape to multi-head format
        3. Apply rotary embeddings if using RoPE
        4. Handle KV cache if installed
        5. Repeat K/V heads to match Q heads
        6. Apply attention (Flash or manual)
        7. Merge heads and project output

        Args:
            x: Input hidden states
            attention_mask: Optional attention mask
            position_offset: Starting position for RoPE (for KV caching)
        """
        batch_size, seq_len, _ = x.shape

        q: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        k: jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads*head_dim"]
        v: jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads*head_dim"]
        q, k, v = self._compute_projections(x)

        q_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        q_heads = self._reshape_queries_to_multihead(q)

        k_heads: jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"]
        k_heads = self._reshape_kv_to_multihead(k)

        v_heads: jaxtyping.Float[torch.Tensor, "batch n_kv_heads seq head_dim"]
        v_heads = self._reshape_kv_to_multihead(v)

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


class MultiHeadLatentAttention(Attention):
    """
    Multi-head Latent Attention (MLA) pattern.

    Used by: DeepSeek-V3.
    Pattern: Compress KV → Project separately → Apply RoPE → Attention

    MLA compresses key-value pairs into a latent space before projection,
    reducing memory usage while maintaining model capacity. It also uses
    separate RoPE dimensions for finer position control.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        kv_compression_dim: int,
        query_compression_dim: int,
        rope_module: "position.PartialRoPE",  # Required for DeepSeek-style models
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
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
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
        keys_content: jaxtyping.Float[torch.Tensor, "batch seq n_heads*head_dim"]
        keys_content = self.key_up(kv_compressed)

        keys_content_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        keys_content_heads = keys_content.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys_content_heads = keys_content_heads.transpose(1, 2)

        # Project to values
        values: jaxtyping.Float[torch.Tensor, "batch seq n_heads*head_dim"]
        values = self.value_up(kv_compressed)

        values_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        values_heads = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values_heads = values_heads.transpose(1, 2)

        # Project to RoPE keys
        keys_rope: jaxtyping.Float[torch.Tensor, "batch seq n_heads*rope_dim"]
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
        queries_content: jaxtyping.Float[torch.Tensor, "batch seq n_heads*head_dim"]
        queries_content = self.query_up(query_compressed)

        queries_content_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        queries_content_heads = queries_content.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        queries_content_heads = queries_content_heads.transpose(1, 2)

        # Project to RoPE queries
        queries_rope: jaxtyping.Float[torch.Tensor, "batch seq n_heads*rope_dim"]
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
    ) -> jaxtyping.Float[torch.Tensor, "batch seq n_heads*head_dim"]:
        """
        Merge attention heads back to single tensor.

        Override: MLA has explicit head_dim, so output is n_heads*head_dim
        which may differ from hidden_dim.
        """
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Transpose heads and sequence dimensions
        x_transposed: jaxtyping.Float[torch.Tensor, "batch seq n_heads head_dim"]
        x_transposed = x.transpose(1, 2).contiguous()

        # Reshape to combine heads
        x_merged: jaxtyping.Float[torch.Tensor, "batch seq n_heads*head_dim"]
        x_merged = x_transposed.view(batch_size, seq_len, self.num_heads * self.head_dim)

        return x_merged

    def _apply_output_projection(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq n_heads*head_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Apply output projection to map back to model dimension."""
        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self.output_proj(x)
        return output

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply Multi-head Latent Attention.

        The process:
        1. Compress inputs to latent spaces
        2. Project to keys, values (content and position separately)
        3. Project queries (content and position separately)
        4. Apply RoPE to position dimensions only
        5. Concatenate content and position features
        6. Apply attention (Flash or manual)
        7. Merge heads and project output

        Note: KV caching is not supported due to compression complexity.
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

        merged: jaxtyping.Float[torch.Tensor, "batch seq n_heads*head_dim"]
        merged = self._merge_heads(attn_output)

        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self._apply_output_projection(merged)

        return output
