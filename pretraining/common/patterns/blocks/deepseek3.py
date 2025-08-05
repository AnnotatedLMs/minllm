# Third Party
import torch.nn as nn

# Project
from pretraining.common.patterns.attention import multi_latent
from pretraining.common.patterns.blocks import core
from pretraining.common.patterns.moe import aux_loss_free
from pretraining.common.patterns.position import rope_partial


class DeepSeek3TransformerBlock(core.PrenormTransformerBlock):
    """
    DeepSeek-V3 specific transformer block.

    Architecture:
    - RMSNorm without bias
    - Multi-head Latent Attention (MLA) with compression
    - Partial RoPE on position-only features
    - Auxiliary-loss-free MoE instead of FFN

    The DeepSeek pattern:
    1. RMSNorm → MLA with PartialRoPE → Residual
    2. RMSNorm → AuxLossFreeMoE → Residual

    Key characteristics:
    - MLA compresses KV to reduce memory (e.g., 2048→512 dims)
    - PartialRoPE separates position from content
    - MoE enables sparse computation at scale
    - No auxiliary loss needed - uses dynamic bias adjustment
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        kv_compression_dim: int,
        query_compression_dim: int,
        rope_module: rope_partial.PartialRoPE,
        rope_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        # DeepSeek uses RMSNorm without bias
        self.attention_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)

        # Multi-head Latent Attention with compression
        self.attention = multi_latent.MultiHeadLatentAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_compression_dim=kv_compression_dim,
            query_compression_dim=query_compression_dim,
            rope_module=rope_module,
            rope_dim=rope_dim,
            dropout=dropout,
            is_causal=True,
            use_flash_attention=use_flash_attention,
        )

        # Auxiliary-loss-free MoE
        self.moe = aux_loss_free.AuxLossFreeMoE(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dropout=dropout,
        )
