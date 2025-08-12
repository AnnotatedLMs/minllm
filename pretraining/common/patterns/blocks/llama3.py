# Standard Library
import typing

# Third Party
from torch import nn

# Project
from pretraining.common.patterns.attention import grouped_query
from pretraining.common.patterns.blocks import core as blocks_core
from pretraining.common.patterns.ffn import gated
from pretraining.common.patterns.position import core as position_core


class Llama3TransformerBlock(blocks_core.PrenormTransformerBlock):
    """
    Llama3-specific transformer block.

    Architecture:
    - RMSNorm without bias (more efficient than LayerNorm)
    - Grouped Query Attention (GQA) for memory efficiency
    - RoPE for position encoding
    - SwiGLU activation in FFN
    - No biases anywhere

    The Llama pattern:
    1. RMSNorm → GQA with RoPE → Residual
    2. RMSNorm → SwiGLU FFN → Residual

    Key characteristics:
    - Designed for efficiency at scale
    - GQA reduces KV cache memory (e.g., 32 query heads, 8 KV heads)
    - SwiGLU provides better performance than GELU
    - RoPE enables long context via position interpolation
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_module: position_core.PrecomputedRoPE,
        norm_eps: float = 1e-5,
        ffn_dim_multiplier: typing.Optional[float] = None,
        multiple_of: int = 256,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        # Llama uses RMSNorm without bias
        self.attention_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)

        # Grouped Query Attention with RoPE
        self.attention = grouped_query.GroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_module=rope_module,
            bias=False,  # No bias in Llama
            is_causal=True,
            use_flash_attention=use_flash_attention,
        )

        # SwiGLU FFN
        self.ffn = gated.MultiplicativeGatedFFN(
            hidden_dim=hidden_dim,
            activation="silu",  # SwiGLU uses SiLU
            bias=False,  # No bias in Llama
            ffn_dim_multiplier=ffn_dim_multiplier,
            multiple_of=multiple_of,
        )
