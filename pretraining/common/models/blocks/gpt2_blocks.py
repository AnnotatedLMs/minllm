# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models.attention import multi_head
from pretraining.common.models.ffn import mlp


class GPT2TransformerBlock(nn.Module):
    """
    GPT-2 transformer block - Pre-norm architecture with biases throughout.
    Radford et al., 2019, https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

    Step-by-step control flow (how modules work together):
    1. LayerNorm: Normalize input, apply learned scale and bias
    2. MultiHeadAttention: Project to Q/K/V, apply causal self-attention
    3. Residual: Add attention output to original input
    4. LayerNorm: Normalize attention output
    5. MLP: Expand 4x with GELU, project back to hidden_dim
    6. Residual: Add MLP output to attention residual

    Learning process (what learns and how):
    - LayerNorm: Learns scale (γ) and bias (β) to normalize and shift distributions
    - MultiHeadAttention: Q/K/V/O projections learn via backprop
    - MLP: Two linear layers learn feature transformations
    - All biases: Provide additional learnable offsets for flexibility

    Key architectural choices:
    - Pre-norm: Stabilizes training by normalizing before transformations
    - GELU: Smoother than ReLU, enables better gradient flow
    - Biases everywhere: More parameters for expressiveness
    - 4x MLP expansion: transformer hidden layer sizing
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: typing.Optional[float] = None,
        max_seq_length: int = 1024,
        norm_eps: float = 1e-5,
        use_flash_attention: bool = True,
        ffn_activation: str = "gelu",
    ):
        super().__init__()

        # GPT-2 uses LayerNorm with bias
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=norm_eps, elementwise_affine=True)

        self.attention = multi_head.MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            max_seq_length=max_seq_length,
            is_causal=True,
            use_flash_attention=use_flash_attention,
        )

        self.ffn = mlp.MLP(
            hidden_dim=hidden_dim,
            intermediate_dim=None,  # Defaults to 4x
            dropout=dropout,
            activation=ffn_activation,
            bias=True,
        )

    def _apply_attention_sublayer(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Apply pre-norm attention sublayer with residual connection."""
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, attention_mask=attention_mask, **kwargs)
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
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply GPT-2 transformer block.

        Two sublayers with pre-normalization:
        1. Attention sublayer: LayerNorm → MHA → Residual
        2. Feedforward sublayer: LayerNorm → MLP → Residual
        """
        x = self._apply_attention_sublayer(x, attention_mask=attention_mask, **kwargs)
        x = self._apply_ffn_sublayer(x)
        return x
