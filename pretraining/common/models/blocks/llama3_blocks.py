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
    Llama 3 transformer block - Efficient architecture with GQA and SwiGLU.
    RoPE: Su et al., 2021, https://arxiv.org/abs/2104.09864
    GQA: Ainslie et al., 2023, https://arxiv.org/pdf/2305.13245
    Llama 3, 2024, https://arxiv.org/pdf/2407.21783

    Step-by-step control flow (how modules work together):
    1. RMSNorm: Normalize by RMS, scale with learned weight (no bias)
    2. GroupedQueryAttention: Project Q/K/V with fewer KV heads
    3. PrecomputedRoPE: Apply rotary embeddings to Q and K
    4. Attention: Compute causal attention with KV head repetition
    5. Residual: Add attention output to original input
    6. RMSNorm: Normalize attention output
    7. SwiGLU: Gate-modulated FFN with element-wise multiplication
    8. Residual: Add SwiGLU output to attention residual

    Learning process (what learns and how):
    - RMSNorm: Learns single scale parameter (no bias for efficiency)
    - GroupedQueryAttention: Q/K/V/O projections learn, K/V share across groups
    - PrecomputedRoPE: NO learning - fixed sinusoidal rotations
    - SwiGLU: Three projections learn (gate, up, down)
    - No biases: Reduces parameters without quality loss

    Key architectural choices:
    - GQA with 8 KV heads: 4x memory reduction vs full MHA
    - RoPE with θ=500,000: Enables 32K+ context windows
    - SwiGLU: Better than ReLU/GELU through learnable gating
    - RMSNorm: Simpler than LayerNorm, equally effective
    - No biases: Cleaner gradients, fewer parameters
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
