# Standard Library
import math
import typing

# Third Party
import jaxtyping
import torch
from torch import nn
from torch.nn import attention as torch_attention
from torch.nn import functional as F


class MultiHeadReshapeMixin:
    """
    Mixin for reshaping tensors between multi-head and standard formats.

    Variation: Split hidden dimension into multiple heads
    Computation: Reshape and transpose operations
    Effect: Enables parallel attention computation across heads
    """

    def _reshape_to_multihead(
        self,
        tensor: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_len head_dim"]:
        """
        Reshape tensor for multi-head attention computation.
        """
        # First reshape to separate head dimension
        reshaped: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads head_dim"]
        reshaped = tensor.view(batch_size, seq_len, num_heads, head_dim)

        # Transpose to put heads before sequence
        multihead: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"]
        multihead = reshaped.transpose(1, 2)

        return multihead

    def _merge_heads(
        self,
        tensor: jaxtyping.Float[torch.Tensor, "batch n_heads seq_len head_dim"],
        hidden_dim: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Merge attention heads back to single tensor.
        """
        batch_size, num_heads, seq_len, head_dim = tensor.shape

        # Transpose heads and sequence dimensions
        x_transposed: jaxtyping.Float[torch.Tensor, "batch seq_len n_heads head_dim"]
        x_transposed = tensor.transpose(1, 2).contiguous()

        # Reshape to combine heads
        x_merged: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        x_merged = x_transposed.view(batch_size, seq_len, hidden_dim)

        return x_merged


class ManualAttentionMixin:
    """
    Mixin that provides manual attention computation (non-Flash path).

    Variation: Explicit attention computation with flexible masking
    Computation: Scores → Masks → Softmax → Dropout → Weighted Values
    Effect: Provides fallback when Flash Attention is not available

    This mixin unifies the manual attention logic from both MultiHeadAttention
    and GroupedQueryAttention, providing flexibility for different masking strategies
    while keeping the core attention computation pure (no projections or output dropout).
    """

    def _compute_attention_scores(
        self,
        q: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"],
        k: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
        head_dim: int,
        attention_scale: typing.Optional[float] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """
        Compute raw attention scores: Q @ K^T / scale.

        Standard scaled dot-product attention - measures how similar each query is to each key.
        Higher scores mean the query and key vectors point in similar directions.
        """
        # Compute dot product
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Apply scaling
        if attention_scale is not None:
            # Use custom scale (e.g., 0.12 from nanogpt)
            scale: float = 1.0 / attention_scale
        else:
            # Standard scaling: 1/sqrt(head_dim)
            scale: float = math.sqrt(head_dim)

        scaled_scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        scaled_scores = scores / scale

        return scaled_scores

    def _apply_buffer_causal_mask(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
        causal_mask: torch.Tensor,
        seq_len: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """
        Apply pre-computed causal mask buffer (MHA style).
        """
        masked_scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        masked_scores = scores.masked_fill(
            causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )
        return masked_scores

    def _create_and_apply_causal_mask(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
        seq_len_q: int,
        seq_len_k: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """
        Create and apply causal mask dynamically (GQA style).
        """
        mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=scores.device),
            diagonal=seq_len_k - seq_len_q + 1,
        )
        masked_scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        masked_scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        return masked_scores

    def _apply_causal_masking(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
        is_causal: bool,
        causal_mask: typing.Optional[torch.Tensor] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """
        Apply causal masking using either pre-computed buffer or dynamic creation.
        """
        if not is_causal:
            return scores

        if causal_mask is not None:
            seq_len = scores.size(2)
            return self._apply_buffer_causal_mask(scores, causal_mask, seq_len)
        else:
            seq_len_q = scores.size(2)
            seq_len_k = scores.size(3)
            return self._create_and_apply_causal_mask(scores, seq_len_q, seq_len_k)

    def _apply_attention_mask(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
        attention_mask: typing.Optional[torch.Tensor],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """
        Apply attention mask to prevent attending to padding tokens.
        """
        if attention_mask is None:
            return scores

        # Reshape mask to broadcast correctly
        # attention_mask is [batch, seq] but scores are [batch, n_heads, seq_q, seq_k]
        # We need to add dimensions for broadcasting
        mask_reshaped = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_k]

        # Apply padding mask (0 positions become -inf)
        masked_scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        masked_scores = scores.masked_fill(mask_reshaped == 0, float("-inf"))

        return masked_scores

    def _compute_attention_weights(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
        attn_dropout: typing.Optional[nn.Dropout] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """
        Convert scores to probabilities and apply dropout.
        """
        attn_probs: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        attn_probs = F.softmax(scores, dim=-1)

        if attn_dropout is not None:
            attn_probs = attn_dropout(attn_probs)

        return attn_probs

    def _apply_attention_to_values(
        self,
        attn_weights: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
        v: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]:
        """
        Apply attention weights to values.
        """
        output: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]
        output = torch.matmul(attn_weights, v)
        return output

    def _compute_manual_attention(
        self,
        q: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"],
        k: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
        v: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
        head_dim: int,
        is_causal: bool = False,
        attention_scale: typing.Optional[float] = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
        causal_mask: typing.Optional[torch.Tensor] = None,
        attn_dropout: typing.Optional[nn.Dropout] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]:
        """
        Compute attention manually (non-Flash path).

        The standard attention process:
        1. Compute attention scores - dot product between queries and keys
        2. Apply causal mask - prevent attending to future positions
        3. Apply attention mask - handle padding tokens
        4. Convert to probabilities - softmax ensures weights sum to 1
        5. Apply dropout - regularization during training
        6. Apply attention to values - weighted average of value vectors

        Output projection and residual dropout are handled by the parent module.
        """
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        scores = self._compute_attention_scores(q, k, head_dim, attention_scale)

        scores = self._apply_causal_masking(scores, is_causal, causal_mask)
        scores = self._apply_attention_mask(scores, attention_mask)

        attn_weights: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        attn_weights = self._compute_attention_weights(scores, attn_dropout)

        output: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]
        output = self._apply_attention_to_values(attn_weights, v)

        return output


class FlashAttentionMixin:
    """
    Mixin that provides Flash Attention functionality for attention mechanisms.

    Variation: Fused kernel attention computation
    Computation: Uses optimized CUDA kernels for memory-efficient attention
    Effect: Reduces memory usage from O(n²) to O(n) and improves speed

    Used by: Both MHA and GQA when flash attention is enabled
    """

    def _apply_flash_attention(
        self,
        q: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"],
        k: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
        v: jaxtyping.Float[torch.Tensor, "batch n_heads seq_k head_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        enable_gqa: bool = False,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]:
        """
        Apply Flash Attention using PyTorch's scaled_dot_product_attention.

        Uses context manager to explicitly request Flash Attention backend.
        Returns only the weighted values (no projection or output dropout).

        Flash Attention v3 does not support dropout. This is a limitation
        of the CUDA kernels. All configs should set dropout=0.0 when using Flash Attention.
        """
        output: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q head_dim"]

        # Try to use Flash Attention with context manager
        with torch_attention.sdpa_kernel(backends=[torch_attention.SDPBackend.FLASH_ATTENTION]):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal and attention_mask is None,
                enable_gqa=enable_gqa,
            )

        return output
