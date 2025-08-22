# Standard Library
import math
import typing

# Third Party
import jaxtyping
import torch
from torch import nn
from torch.nn import attention as torch_attention
from torch.nn import functional as F

try:
    # Third Party
    from flash_mla import flash_attn_varlen_qkvpacked_func

    FLASH_MLA_AVAILABLE = True
except ImportError:
    FLASH_MLA_AVAILABLE = False


class ManualSDPAMixin:
    """
    Mixin that provides manual scaled dot-product attention (SDPA) computation.

    Scholarship:
    Attention Is All You Need, 2017, https://arxiv.org/pdf/1706.03762, 3.2

    Significance:
    Enables tokens to attend to each other based on learned similarities.
    Scaling prevents gradients from vanishing when dot products get large.

    Init:
    This mixin has no initialization or learnable parameters.
    It provides computational methods used by attention modules.

    Step-by-step control flow (_compute_manual_attention):
    1. Compute attention scores via dot product between Q and K
    2. Scale scores (by 1/sqrt(head_dim) or custom attention_scale)
    3. Apply causal mask to prevent attending to future tokens
    4. Apply padding mask to ignore padding tokens
    5. Convert scores to probabilities via softmax
    6. Apply dropout for regularization (during training)
    7. Weight values by attention probabilities

    Learning/Optimization process:
    - This mixin contains no learnable parameters.
    - It implements the mathematical operations of attention: QK^T/√d followed by softmax and weighted sum.
    - The scaling factor can be customized via attention_scale parameter (e.g., for YaRN adjustments).

    - Purpose in architecture:
      - Computes similarity between queries and keys to find relevant tokens
      - Softmax ensures attention weights sum to 1, creating a weighted average
      - Enables Q/K/V projections (in parent classes) to learn what features matter
      - Result: Allows model to dynamically select which tokens to use for predictions"""

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


class FlashMLAMixin:
    """
    Mixin that provides FlashMLA optimized kernel for Multi-head Latent Attention.

    Scholarship:
    FlashMLA Deep-Dive, 2025, https://github.com/deepseek-ai/FlashMLA
    FlashAttention-3, 2024, https://arxiv.org/abs/2407.08608

    Significance:
    Achieves up to 660 TFlops in compute-bound MLA decoding scenarios.
    Uses "seesaw" scheduling to overlap CUDA Core and Tensor Core operations with one output matrix.

    Init:
    This mixin has no initialization or learnable parameters.
    It wraps FlashMLA's optimized CUDA kernels for SM90/SM100 GPUs.

    Step-by-step control flow (_compute_flash_mla_attention):
    1. Reshape tensors from batch format to varlen format
    2. Pack Q, K, V into single tensor for kernel efficiency
    3. Create cumulative sequence lengths for batch handling
    4. Call optimized FlashMLA kernel with packed data
    5. Reshape output back to standard batch format

    Learning/Optimization process:
    - This mixin contains no learnable parameters.
    - It implements the same mathematical operations as ManualSDPAMixin but with optimized kernels.

    - Kernel optimizations (compute-bound scenario):
      - Seesaw scheduling: Alternates operations between two warpgroups with one output matrix
      - Fine-grained TMA copy-GEMM pipelining: Overlaps memory access with computation
      - Achieves 80% Tensor Core utilization on H800 GPUs

    - Purpose in architecture:
      - Provides faster training and inference for MLA models
      - Reduces memory usage through online softmax computation
      - Enables efficient handling of compressed attention dimensions
    """

    @staticmethod
    def is_available() -> bool:
        """Check if FlashMLA is available."""
        if not FLASH_MLA_AVAILABLE:
            return False

        # Check GPU capability (needs SM90 or SM100)
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            return capability in [(9, 0), (10, 0)]
        return False

    def _compute_flash_mla_attention(
        self,
        queries_full: jaxtyping.Float[torch.Tensor, "batch heads seq total_dim"],
        keys_full: jaxtyping.Float[torch.Tensor, "batch heads seq total_dim"],
        values: jaxtyping.Float[torch.Tensor, "batch heads seq head_dim"],
        total_dim: int,
        is_causal: bool = True,
        attention_scale: typing.Optional[float] = None,
        attn_dropout: typing.Optional[nn.Dropout] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch heads seq head_dim"]:
        """
        Compute MLA attention using FlashMLA kernels (training mode).

        Args:
            queries_full: Queries with content and position features [batch, heads, seq, total_dim]
            keys_full: Keys with content and position features [batch, heads, seq, total_dim]
            values: Value vectors [batch, heads, seq, head_dim]
            total_dim: Total dimension (head_dim + rope_dim)
            is_causal: Whether to apply causal masking
            attention_scale: Custom attention scale (includes mscale for YaRN)
            attn_dropout: Dropout module (not supported in FlashMLA)

        Returns:
            Attention output [batch, heads, seq, head_dim]
        """
        if not FLASH_MLA_AVAILABLE:
            raise RuntimeError(
                "FlashMLA is not available. Please install it with: "
                "pip install git+https://github.com/deepseek-ai/FlashMLA.git"
            )
        if attn_dropout != 0.0:
            raise RuntimeError("FlashMLA does not support dropout.")

        batch_size, num_heads, seq_len, _ = queries_full.shape
        head_dim = values.shape[-1]

        # FlashMLA expects varlen format: [total_tokens, num_heads, dim]
        # Reshape from [batch, heads, seq, dim] to [batch * seq, heads, dim]
        queries_flat = queries_full.transpose(1, 2).reshape(
            batch_size * seq_len, num_heads, total_dim
        )
        keys_flat = keys_full.transpose(1, 2).reshape(batch_size * seq_len, num_heads, total_dim)
        values_flat = values.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_dim)

        # Pack QKV for FlashMLA (it expects concatenated format)
        # [batch * seq, heads, total_dim + total_dim + head_dim]
        qkv_packed = torch.cat([queries_flat, keys_flat, values_flat], dim=-1)

        # Create cumulative sequence lengths for varlen format
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=queries_full.device
        )

        # Compute softmax scale if not provided
        if attention_scale is None:
            attention_scale = 1.0 / math.sqrt(total_dim)

        # Call FlashMLA varlen function
        output, lse = flash_attn_varlen_qkvpacked_func(
            qkv=qkv_packed,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
            head_dim_qk=total_dim,  # Q/K dimension (includes RoPE)
            dropout_p=attn_dropout,
            softmax_scale=attention_scale,
            causal=is_causal,
            deterministic=False,
            is_varlen=True,
        )

        # Reshape output back to [batch, heads, seq, head_dim]
        output = output.reshape(batch_size, seq_len, num_heads, head_dim)
        output = output.transpose(1, 2).contiguous()

        return output
