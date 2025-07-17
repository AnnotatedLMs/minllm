# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
# Local
from pretraining.common.base.models import transformer
from pretraining.common.patterns import attention
from pretraining.common.patterns import ffn
from pretraining.common.patterns import moe
from pretraining.common.patterns.components import position
from pretraining.configs.transformer import attention_configs

# TODO abstract methods for transformer block? (ie _apply_attention_sublayer, _apply_ffn_sublayer)


class TransformerBlock(transformer.BaseTransformerBlock):
    """
    Base class for transformer patterns with common implementations.

    This class provides standard operations shared across transformer
    block patterns. Subclasses can override specific methods.
    """

    def __init__(self):
        super().__init__()

    def _apply_residual_connection(
        self,
        residual: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Apply residual connection."""
        combined: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        combined = residual + output
        return combined


class BiasedLNTransformerBlock(TransformerBlock):
    """
    Transformer block using LayerNorm with bias.

    Used by: GPT-2.
    Pattern: x → LayerNorm(bias=True) → Attention → Add(residual) → LayerNorm(bias=True) → FFN → Add(residual)

    Characteristics:
    - Uses LayerNorm with learnable bias and scale parameters
    - Includes bias in all linear projections
    - Pre-layer normalization (norm before sublayer operations)
    - Combined QKV projection in attention
    - GELU activation in FFN
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        max_seq_length: int = 1024,
        activation: str = "gelu",
        norm_eps: float = 1e-5,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        # Initialize normalization layers
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=norm_eps, elementwise_affine=bias)
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=norm_eps, bias=bias)

        self.attn = attention.MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            max_seq_length=max_seq_length,
            is_causal=True,
            use_flash_attention=use_flash_attention,
        )

        self.ffn = ffn.MLP(
            hidden_dim=hidden_dim,
            intermediate_dim=None,  # Will default to 4x
            dropout=dropout,
            activation=activation,
            bias=bias,
        )

    def _apply_attention_norm(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply layer normalization before attention.

        GPT-2 specific: Norm is applied to input before operation.
        """
        normed: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        normed = self.ln_1(x)
        return normed

    def _apply_ffn_norm(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Apply layer normalization before FFN."""
        normed: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        normed = self.ln_2(x)
        return normed

    def _apply_attention_sublayer(
        self,
        residual: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply attention sublayer with biased LayerNorm.

        GPT-2 applies LayerNorm before the attention, then adds the residual.
        """
        normed_x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        normed_x = self._apply_attention_norm(residual)

        attn_out: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        attn_out = self.attn(
            normed_x,
            attention_mask=attention_mask,
        )

        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self._apply_residual_connection(residual, attn_out)

        return output

    def _apply_ffn_sublayer(
        self, residual: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply feedforward sublayer with biased LayerNorm.

        Same pattern: LayerNorm → FFN → Add(residual).
        """
        normed_x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        normed_x = self._apply_ffn_norm(residual)

        ffn_out: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        ffn_out = self.ffn(normed_x)

        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self._apply_residual_connection(residual, ffn_out)

        return output

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply GPT-2 style transformer block with biased LayerNorm.

        The pattern:
        1. Apply attention sublayer (LayerNorm → attention → residual)
        2. Apply FFN sublayer (LayerNorm → FFN → residual)
        """

        attn_output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        attn_output = self._apply_attention_sublayer(
            residual=x,
            attention_mask=attention_mask,
        )

        final_output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        final_output = self._apply_ffn_sublayer(residual=attn_output)

        return final_output


class RMSNormTransformerBlock(TransformerBlock):
    """
    Transformer block using RMSNorm without bias.

    Used by: Llama 3, DeepSeek-V3, and most modern LLMs.
    Pattern: x → RMSNorm → Attention → Add(residual) → RMSNorm → FFN/MoE → Add(residual)

    Characteristics:
    - Uses RMSNorm (Root Mean Square Layer Normalization) for efficiency
    - No bias in linear projections (bias=False)
    - Pre-layer normalization (norm before sublayer operations)
    - Supports Grouped Query Attention (GQA) with separate num_kv_heads
    - Supports SwiGLU activation in FFN
    - Can use MoE instead of FFN (DeepSeek-V3)
    - Uses Rotary Position Embeddings (RoPE) instead of learned position embeddings
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: typing.Optional[int] = None,
        rope_module: typing.Optional[
            typing.Union[position.PrecomputedRoPE, position.PartialRoPE]
        ] = None,
        dropout: float = 0.0,
        bias: bool = False,
        norm_eps: float = 1e-5,
        rope_dim: int = 64,
        activation: str = "silu",
        ffn_dim_multiplier: typing.Optional[float] = None,
        multiple_of: int = 256,
        # For MoE variants
        use_moe: bool = False,
        num_experts: typing.Optional[int] = None,
        num_experts_per_token: typing.Optional[int] = None,
        # For MLA variants
        use_mla: bool = False,
        mla_config: typing.Optional[attention_configs.MultiHeadLatentAttentionConfig] = None,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        self.input_layernorm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_dim, eps=norm_eps)

        # Initialize attention based on architecture
        if use_mla and mla_config is not None:
            # MLA for DeepSeek
            if rope_module is None:
                raise ValueError("MultiHeadLatentAttention requires rope_module")
            self.attention = attention.MultiHeadLatentAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=mla_config.head_dim,
                kv_compression_dim=mla_config.kv_compression_dim,
                query_compression_dim=mla_config.query_compression_dim,
                rope_module=rope_module,
                rope_dim=rope_dim,
                dropout=dropout,
                is_causal=True,
                use_flash_attention=use_flash_attention,
            )
        elif num_kv_heads is not None and num_kv_heads != num_heads:
            # GQA for Llama
            if rope_module is None:
                raise ValueError("GroupedQueryAttention requires rope_module")
            self.attention = attention.GroupedQueryAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                rope_module=rope_module,
                dropout=dropout,
                bias=bias,
                is_causal=True,
                use_flash_attention=use_flash_attention,
            )
        else:
            # Regular MHA for GPT-2 (doesn't use RoPE)
            self.attention = attention.MultiHeadAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                is_causal=True,
                use_flash_attention=use_flash_attention,
            )

        # Initialize FFN or MoE
        if use_moe and num_experts is not None and num_experts_per_token is not None:
            self.moe = moe.AuxLossFreeMoE(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                dropout=dropout,
            )
        else:
            # Use regular FFN - SwiGLU for Llama-style
            if activation == "silu":
                self.ffn = ffn.MultiplicativeGatedFFN(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                    bias=bias,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    multiple_of=multiple_of,
                )
            else:
                self.ffn = ffn.MLP(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                    bias=bias,
                )

    def _normalize_for_attention(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply normalization before attention.

        Modern models use RMSNorm for efficiency, but the pattern
        is the same regardless of norm type.
        """
        normed: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        normed = self.input_layernorm(x)
        return normed

    def _normalize_for_ffn(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Apply normalization before feedforward network."""
        normed: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        normed = self.post_attention_layernorm(x)
        return normed

    def _apply_attention_sublayer(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply attention with RMSNorm and residual connection.

        Modern models normalize before the operation for training stability.
        """
        residual: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        residual = x

        normed: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        normed = self._normalize_for_attention(x)

        attn_out: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        # Check if this is GroupedQueryAttention which supports position_offset
        if isinstance(self.attention, attention.GroupedQueryAttention):
            attn_out = self.attention(
                normed,
                attention_mask=attention_mask,
                position_offset=position_offset,
            )
        else:
            # MultiHeadAttention doesn't support position_offset yet
            attn_out = self.attention(
                normed,
                attention_mask=attention_mask,
            )

        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self._apply_residual_connection(residual, attn_out)
        return output

    def _apply_ffn_sublayer(
        self, residual: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply feedforward network with pre-normalization and residual connection.

        For models with MoE, the FFN would be replaced with the MoE layer.
        """
        # Apply normalization
        normed: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        normed = self._normalize_for_ffn(residual)

        # Step 3: Apply feedforward network (or MoE)
        ffn_out: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        if hasattr(self, "ffn"):
            ffn_out = self.ffn(normed)
        elif hasattr(self, "moe"):
            ffn_out = self.moe(normed)
        else:
            raise AttributeError(
                f"{self.__class__.__name__} has no feedforward component (ffn or moe)"
            )

        # Step 4: Add residual connection
        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = self._apply_residual_connection(residual, ffn_out)
        return output

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply modern transformer block with RMSNorm.

        The pattern:
        1. Apply attention sublayer (RMSNorm → attention → residual)
        2. Apply FFN/MoE sublayer (RMSNorm → FFN/MoE → residual)

        This provides better training stability for deep networks compared
        to the older LayerNorm with bias approach.
        """
        after_attention: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        after_attention = self._apply_attention_sublayer(
            x,
            attention_mask=attention_mask,
            position_offset=position_offset,
        )

        final_output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        final_output = self._apply_ffn_sublayer(residual=after_attention)

        return final_output
