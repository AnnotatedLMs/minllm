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
from pretraining.common.base import attention


class Attention(attention.BaseAttention):
    """
    Educational base class showing common attention patterns.

    Most attention mechanisms share the same core operations:
    1. Project inputs to queries, keys, and values - creates three different views of the same input:
       - Queries (Q): "what information am I looking for?"
       - Keys (K): "what information do I contain?"
       - Values (V): "what information should I output if selected?"
    2. Compute attention scores between queries and keys - calculates dot product similarity
       between each query and all keys to determine which positions are most relevant
    3. Apply masks to prevent attending to certain positions - in autoregressive models,
       masks future positions to ensure predictions only depend on past context
    4. Normalize scores to get attention weights - applies softmax to create a probability
       distribution where all weights sum to 1, forcing the model to prioritize
    5. Use weights to combine values - performs weighted average of value vectors based on
       attention weights, aggregating information from relevant positions

    The variations between papers typically involve:
    - How queries, keys, and values are computed (separate projections vs combined)
    - Whether position information is added via embeddings or RoPE
    - How many key-value pairs each query attends to (full attention vs grouped/compressed)
    - Additional operations like compression or gating

    See the forward() method in each implementation for the specific flow.
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

        Standard scaled dot-product attention - measures how similar each query is to each key.
        Higher scores mean the query and key vectors point in similar directions.
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

        For autoregressive models - position i can only attend to positions 0...i.
        """
        mask: jaxtyping.Bool[torch.Tensor, "seq seq"]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask

    def _apply_causal_mask(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
        seq_len: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """Apply causal mask to attention scores - prevents attending to future positions."""
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
        """Apply optional attention mask (e.g., for padding) - prevents attending to pad tokens."""
        if attention_mask is None:
            return scores

        # Reshape mask to broadcast correctly
        # attention_mask is [batch, seq] but scores are [batch, n_heads, seq_q, seq_k]
        # We need to add dimensions and broadcast
        mask_reshaped = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_k]

        # Apply padding mask
        masked_scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        masked_scores = scores.masked_fill(mask_reshaped == 0, float("-inf"))

        return masked_scores

    def _compute_attention_probs(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """Convert attention scores to probabilities - ensures weights sum to 1 across keys."""
        attn_probs: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]
        attn_probs = F.softmax(scores, dim=-1)
        return attn_probs

    def _apply_attention_dropout(
        self,
        attn_probs: jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"],
    ) -> jaxtyping.Float[torch.Tensor, "batch n_heads seq_q seq_k"]:
        """Apply dropout to attention probabilities - randomly zeros some attention weights during training."""
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

        Important: Flash Attention does not support dropout. This is a limitation
        of the CUDA kernels. Modern LLMs typically don't use dropout anyway due
        to model scale and computational efficiency requirements. All configs
        should set dropout=0.0 when using Flash Attention.
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

        The standard attention process:
        1. Compute attention scores - dot product between queries and keys
        2. Apply causal mask - set future positions to -inf so they get 0 weight after softmax
        3. Apply attention mask - set padded positions to -inf
        4. Convert to probabilities - softmax ensures weights sum to 1 for each query
        5. Apply dropout - randomly zero weights during training for regularization
        6. Apply attention to values - weighted average of value vectors
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
