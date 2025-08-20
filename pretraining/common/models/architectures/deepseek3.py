# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.mixins import architecture_mixins
from pretraining.common.mixins import fsdp_mixins
from pretraining.common.mixins import generation_mixins
from pretraining.common.models import outputs
from pretraining.common.models.blocks import deepseek3_blocks
from pretraining.common.models.heads import multi_token
from pretraining.common.models.position import partial_rope
from pretraining.configs.model.architectures import deepseek
from pretraining.configs.model.components import position


class DeepSeek3(
    nn.Module,
    architecture_mixins.TransformerBlockStackMixin,
    generation_mixins.AutoregressiveGenerationMixin,
    fsdp_mixins.FSDPMixin,
):
    """
    DeepSeek-V3 Language Model implementation.

    Used by: DeepSeek-V3

    Variation: Multi-head Latent Attention (MLA) with PartialRoPE
    Computation: Compresses Q/K/V through bottleneck, applies RoPE to position-only features
    Effect: Reduces activation memory while separating content and position processing
    """

    def __init__(
        self,
        # Core dimensions
        vocab_size: int,
        hidden_dim: int,
        n_layers: int,
        block_size: int,
        # RoPE params
        rope_theta: float = 10000.0,
        rope_dim: int = 64,
        # MLA attention params
        num_heads: int = 128,
        head_dim: int = 128,
        kv_compression_dim: int = 512,
        query_compression_dim: int = 1536,
        # MoE params
        num_experts: int = 256,
        num_experts_per_token: int = 6,
        intermediate_dim: typing.Optional[int] = None,
        n_shared_experts: int = 2,
        shared_expert_ratio: float = 0.1,
        activation: str = "silu",
        # Other params
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        lm_head_bias: bool = False,
        use_flash_attention: bool = False,
        # Multi-token prediction params
        mtp_n_predict: int = 3,
        mtp_dropout: typing.Optional[float] = None,
        # MoE auxiliary loss params
        aux_loss_alpha: float = 0.001,
        bias_update_speed: float = 0.001,
    ):
        super().__init__()

        # Store dimensions
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Token embeddings
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
        )

        # Dropout for embeddings
        self.embedding_dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Key RoPE: Will have YaRN applied during context extension
        self.key_rope = partial_rope.PartialRoPE(
            dim=rope_dim,
            theta=rope_theta,
        )

        # Query RoPE: No YaRN, stays at original frequencies
        self.query_rope = partial_rope.PartialRoPE(
            dim=rope_dim,
            theta=rope_theta,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                deepseek3_blocks.DeepSeek3TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    kv_compression_dim=kv_compression_dim,
                    query_compression_dim=query_compression_dim,
                    key_rope_module=self.key_rope,
                    query_rope_module=self.query_rope,
                    rope_dim=rope_dim,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    intermediate_dim=intermediate_dim,
                    n_shared_experts=n_shared_experts,
                    shared_expert_ratio=shared_expert_ratio,
                    activation=activation,
                    dropout=dropout,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                    aux_loss_alpha=aux_loss_alpha,
                    bias_update_speed=bias_update_speed,
                )
                for _ in range(n_layers)
            ]
        )

        # Final RMSNorm
        self.norm = nn.RMSNorm(hidden_dim, eps=norm_eps)

        # Output projection - separate (not tied)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=lm_head_bias)

        # Multi-token prediction heads
        self.mtp_heads = multi_token.MultiTokenPredictionHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            depth=mtp_n_predict,
            dropout=mtp_dropout,
        )

    @classmethod
    def from_config(cls, config: deepseek.DeepSeek3Config) -> "DeepSeek3":
        """Create DeepSeek3 from a config object."""

        attn_config = config.transformer.attention
        moe_config = config.transformer.moe

        return cls(
            # Core dimensions
            vocab_size=config.vocab_size,
            hidden_dim=config.transformer.hidden_dim,
            n_layers=config.transformer.n_layers,
            block_size=config.transformer.block_size,
            # RoPE params
            rope_theta=config.transformer.rope.theta,
            rope_dim=attn_config.rope_dim,
            # MLA attention params
            num_heads=attn_config.num_heads,
            head_dim=attn_config.head_dim,
            kv_compression_dim=attn_config.kv_compression_dim,
            query_compression_dim=attn_config.query_compression_dim,
            use_flash_attention=config.transformer.attention.use_flash_attention,
            # MoE params
            num_experts=moe_config.num_experts,
            num_experts_per_token=moe_config.num_experts_per_token,
            intermediate_dim=moe_config.intermediate_dim,
            n_shared_experts=moe_config.n_shared_experts,
            shared_expert_ratio=moe_config.shared_expert_ratio,
            activation=moe_config.activation,
            # Other params
            dropout=config.transformer.dropout,
            norm_eps=config.transformer.normalization.norm_eps,
            lm_head_bias=config.output_head.lm_head_bias,
            # MTP params
            mtp_n_predict=config.mtp.n_predict,
            mtp_dropout=config.mtp.dropout,
            # MoE auxiliary loss params
            aux_loss_alpha=moe_config.aux_loss_alpha,
            bias_update_speed=moe_config.bias_update_speed,
        )

    def forward(
        self,
        input_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> outputs.ForwardOutput:
        """Forward pass for DeepSeek3."""
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        x = self.token_embeddings(input_ids)

        if self.embedding_dropout is not None and self.training:
            x = self.embedding_dropout(x)

        aux_losses = []
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"] = x

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

            if self.training:
                aux_loss = block.moe.get_auxiliary_loss()
                if aux_loss is not None:
                    aux_losses.append(aux_loss)

        hidden_states = self.norm(hidden_states)

        logits: jaxtyping.Float[torch.Tensor, "batch seq_len vocab"]
        logits = self.lm_head(hidden_states)

        # Compute MTP logits during training
        mtp_logits = None
        if self.training:
            mtp_logits = self.mtp_heads(hidden_states)

        return outputs.ForwardOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            aux_losses=aux_losses if aux_losses else None,
        )

    def generate(
        self,
        token_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: typing.Optional[int] = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> jaxtyping.Int[torch.Tensor, "batch new_seq"]:
        """Generate tokens autoregressively."""
        # Use the mixin's generate method, passing our forward as model_forward
        return super().generate(
            model_forward=self.forward,
            block_size=self.block_size,
            token_ids=token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            attention_mask=attention_mask,
        )

    # FSDP methods
    def get_fsdp_wrappable_modules(self) -> typing.Set[typing.Type[nn.Module]]:
        """Get module types that should be wrapped by FSDP."""
        return {deepseek3_blocks.DeepSeek3TransformerBlock}

    def get_transformer_blocks(self) -> typing.List[nn.Module]:
        """Get transformer blocks for FSDP wrapping."""
        return list(self.blocks)

    def get_fsdp_special_modules(self) -> typing.Dict[str, nn.Module]:
        """Get special modules that need specific FSDP handling."""
        special_modules = {}

        special_modules["token_embeddings"] = self.token_embeddings
        special_modules["lm_head"] = self.lm_head

        # MTP heads can be large with multiple prediction steps
        if self.mtp_heads is not None:
            special_modules["mtp_heads"] = self.mtp_heads

        return special_modules

    def update_for_context_extension(
        self,
        new_max_seq_len: int,
        yarn_config: position.YaRNConfig,
    ) -> None:
        """
        Trigger context extension for YaRN scaling.

        Per DeepSeek-V3 paper: "YaRN... being applied exclusively to the
        decoupled shared key k_R_t"

        Only updates key RoPE module, query RoPE stays unchanged.

        Two-phase extension (DeepSeek-V3):
        - Phase 1: 4K → 32K (scale_factor=8)
        - Phase 2: 32K → 128K (scale_factor=32)

        Args:
            new_max_seq_len: New maximum sequence length
            yarn_config: YaRN configuration for this phase
        """
        # Store YaRN config in key RoPE module
        self.key_rope.yarn_config = yarn_config

        # Apply YaRN scaling to key RoPE only
        self.key_rope.update_for_context_extension(
            new_max_seq_len=new_max_seq_len,
            original_context_len=yarn_config.original_context_len,
            beta_fast=yarn_config.beta_fast,
            beta_slow=yarn_config.beta_slow,
            extrapolation_factor=yarn_config.extrapolation_factor,
            attn_factor=yarn_config.attn_factor,
            mscale_all_dim=yarn_config.mscale_all_dim,
        )
