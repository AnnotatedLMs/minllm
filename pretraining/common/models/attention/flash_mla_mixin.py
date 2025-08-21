# Standard Library
import math
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Try to import FlashMLA - will be optional
try:
    # Third Party
    from flash_mla import flash_attn_varlen_qkvpacked_func

    FLASH_MLA_AVAILABLE = True
except ImportError:
    FLASH_MLA_AVAILABLE = False


class FlashMLAMixin:
    """
    Mixin that provides FlashMLA functionality for Multi-head Latent Attention during training.

    FlashMLA is an optimized CUDA kernel implementation of MLA that provides:
    - Memory-efficient attention computation
    - Faster performance on compatible GPUs (SM90/SM100)
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
