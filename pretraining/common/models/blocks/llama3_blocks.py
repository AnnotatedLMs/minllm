# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models.attention import grouped_query
from pretraining.common.models.ffn import swiglu
from pretraining.common.models.position import rope


class Llama3TransformerBlock(nn.Module):
    """
    Llama 3 specific transformer block.

    Architecture:
    - RMSNorm without bias (more efficient than LayerNorm)
    - Grouped Query Attention (GQA) for memory efficiency
    - RoPE for position encoding
    - SwiGLU activation in FFN
    - No biases anywhere

    The Llama 3 pattern:
    1. RMSNorm → GQA with RoPE → Residual
    2. RMSNorm → SwiGLU FFN → Residual

    Key characteristics:
    - Designed for efficiency at scale
    - GQA reduces KV cache memory (e.g., 32 query heads, 8 KV heads)
    - SwiGLU provides better performance than GELU
    - RoPE enables long context via position interpolation
    - No position embeddings at block level (RoPE applied during attention)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_module: rope.PrecomputedRoPE,
        dropout: typing.Optional[float] = None,
        norm_eps: float = 1e-5,
        ffn_activation: str = "silu",
        ffn_dim_multiplier: typing.Optional[float] = None,
        multiple_of: int = 256,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        self.attention_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)

        self.attention = grouped_query.GroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_module=rope_module,
            dropout=dropout,
            bias=False,  # No bias in Llama 3
            is_causal=True,
            use_flash_attention=use_flash_attention,
        )

        self.ffn = swiglu.SwiGLU(
            hidden_dim=hidden_dim,
            intermediate_dim=None,  # Will be calculated based on Llama 3 formula
            dropout=dropout,
            activation=ffn_activation,
            bias=False,
            ffn_dim_multiplier=ffn_dim_multiplier,
            multiple_of=multiple_of,
        )

    def _apply_attention_sublayer(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply pre-norm attention sublayer with residual connection.

        Includes position_offset for RoPE to handle cached generation.
        """
        residual = x
        x = self.attention_norm(x)
        x = self.attention(
            x, attention_mask=attention_mask, position_offset=position_offset, **kwargs
        )
        return residual + x

    def _apply_ffn_sublayer(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Apply pre-norm FFN sublayer with residual connection."""
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return residual + x

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply Llama 3 transformer block.

        Two sublayers with pre-normalization:
        1. Attention sublayer: RMSNorm → GQA(with RoPE) → Residual
        2. Feedforward sublayer: RMSNorm → SwiGLU → Residual

        Args:
            x: Input tensor
            attention_mask: Optional attention mask for padding
            position_offset: Position offset for RoPE (used during generation with KV cache)
        """
        x = self._apply_attention_sublayer(
            x, attention_mask=attention_mask, position_offset=position_offset, **kwargs
        )
        x = self._apply_ffn_sublayer(x)
        return x
