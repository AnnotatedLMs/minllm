# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models.attention import multi_latent
from pretraining.common.models.moe import aux_loss_free
from pretraining.common.models.position import partial_rope


class DeepSeek3TransformerBlock(nn.Module):
    """
    DeepSeek-V3 transformer block - MLA compression with auxiliary-loss-free MoE.
    DeepSeekMoE, 2024, https://arxiv.org/pdf/2401.06066
    Deepseek-V2, 2024, https://arxiv.org/pdf/2405.04434
    Deepseek-V3, 2025, https://arxiv.org/pdf/2412.19437

    Step-by-step control flow (how modules work together):
    1. RMSNorm: Normalize input with learned scale (no bias)
    2. MultiHeadLatentAttention: Compress to latent, expand to Q/K/V
    3. PartialRoPE: Apply rotation only to position features
    4. Attention: Compute with separated content and position
    5. Residual: Add attention output to original input
    6. RMSNorm: Normalize for MoE layer
    7. AuxLossFreeMoE: Route tokens to experts with bias-based balancing
    8. Residual: Add MoE output to attention residual

    Learning process (what learns and how):
    - RMSNorm: Learns scale parameter for normalization
    - MLA: Compression and expansion projections learn via backprop
    - PartialRoPE: NO learning - fixed rotations on position dims
    - AuxLossFreeMoE: Expert centroids and expert networks learn
    - Load balancing: Bias adjustment without gradients (auxiliary-loss-free)

    Key architectural choices:
    - MLA: 4x compression reduces attention memory and compute
    - Partial RoPE: Separates content/position for better compression
    - Aux-loss-free MoE: Load balance without disrupting gradients
    - Shared experts: Capture common patterns, routed experts specialize
    - No biases: Cleaner optimization, fewer parameters
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        kv_compression_dim: int,
        query_compression_dim: int,
        key_rope_rotation: partial_rope.DecoupledRoPE,  # Rotation module for keys (with YaRN)
        query_rope_rotation: partial_rope.DecoupledRoPE,  # Rotation module for queries (no YaRN)
        rope_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        intermediate_dim: typing.Optional[int] = None,
        n_shared_experts: int = 2,
        shared_expert_ratio: float = 0.1,
        activation: str = "silu",
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        use_flash_attention: bool = False,
        aux_loss_alpha: float = 0.001,
        bias_update_speed: float = 0.001,
    ):
        super().__init__()

        self.attention_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)

        self.attention = multi_latent.MultiHeadLatentAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_compression_dim=kv_compression_dim,
            query_compression_dim=query_compression_dim,
            key_rope_rotation=key_rope_rotation,
            query_rope_rotation=query_rope_rotation,
            rope_dim=rope_dim,
            dropout=dropout,
            is_causal=True,
            use_flash_attention=use_flash_attention,
        )

        self.moe = aux_loss_free.AuxLossFreeMoE(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            intermediate_dim=intermediate_dim,
            n_shared_experts=n_shared_experts,
            shared_expert_ratio=shared_expert_ratio,
            activation=activation,
            dropout=dropout,
            aux_loss_alpha=aux_loss_alpha,
            bias_update_speed=bias_update_speed,
        )

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Forward pass through DeepSeek3 transformer block.

        Pre-norm architecture:
        1. RMSNorm → MLA → Residual
        2. RMSNorm → MoE → Residual
        """
        # Attention with residual
        attn_input: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        attn_input = self.attention_norm(x)

        attn_output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        attn_output = self.attention(attn_input, attention_mask)

        x = x + attn_output

        # MoE with residual
        moe_input: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        moe_input = self.ffn_norm(x)

        moe_output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        moe_output = self.moe(moe_input)

        x = x + moe_output

        return x
