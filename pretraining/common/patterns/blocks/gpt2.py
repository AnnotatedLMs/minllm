# Third Party
from torch import nn

# Project
from pretraining.common.patterns.attention import multi_head
from pretraining.common.patterns.blocks import core
from pretraining.common.patterns.ffn import mlp


class GPT2TransformerBlock(core.PrenormTransformerBlock):
    """
    GPT-2 specific transformer block.

    Architecture:
    - LayerNorm with bias (learnable scale and shift)
    - Multi-Head Attention with bias in all projections
    - MLP with GELU activation and 4x expansion

    The GPT-2 pattern:
    1. LayerNorm(bias=True) → MHA(bias=True) → Residual
    2. LayerNorm(bias=True) → MLP(GELU, bias=True) → Residual

    Key characteristics:
    - Uses biases throughout for additional learnable parameters
    - GELU activation for smoother gradients
    - Standard 4x hidden dimension expansion in MLP
    - No position embeddings in the block itself (handled at model level)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_length: int = 1024,
        norm_eps: float = 1e-5,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        # GPT-2 uses LayerNorm with bias
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=norm_eps, bias=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=norm_eps, bias=True)

        # Multi-head attention with bias
        self.attention = multi_head.MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,  # GPT-2 uses bias
            max_seq_length=max_seq_length,
            is_causal=True,
            use_flash_attention=use_flash_attention,
        )

        # MLP with GELU activation
        self.ffn = mlp.MLP(
            hidden_dim=hidden_dim,
            intermediate_dim=None,  # Defaults to 4x
            dropout=dropout,
            activation="gelu",  # GPT-2 specific
            bias=True,  # GPT-2 uses bias
        )
