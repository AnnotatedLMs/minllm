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
from pretraining.common.models.blocks import llama3_blocks
from pretraining.common.models.position import rope
from pretraining.configs.model.architectures import llama


class Llama3(
    nn.Module,
    architecture_mixins.TransformerBlockStackMixin,
    fsdp_mixins.FSDPMixin,
    generation_mixins.CachedGenerationMixin,
):
    """
    Llama 3 Language Model implementation.

    Used by: Llama 3 family

    Variation: Token embeddings only with RoPE applied in attention
    Computation: No position embeddings added to input, RoPE rotates Q/K vectors
    Effect: Model learns relative positions that generalize to any sequence length

    Variation: RMSNorm without bias, GQA, and SwiGLU
    Computation: Simpler normalization, shared KV heads, gated FFN
    Effect: Reduces memory and computation while maintaining model quality

    Variation: No bias terms in any linear layers
    Computation: All nn.Linear layers have bias=False
    Effect: Improves training stability and slightly reduces parameters

    Variation: Separate lm_head (no weight tying)
    Computation: Independent output projection matrix
    Effect: More parameters but often better performance at scale
    """

    def __init__(
        self,
        # Core dimensions
        vocab_size: int,
        hidden_dim: int,
        n_layers: int,
        block_size: int,
        # Transformer params
        num_heads: int,  # query heads
        num_kv_heads: int,  # key/value heads for GQA
        dropout: typing.Optional[float] = None,
        norm_eps: float = 1e-5,
        use_flash_attention: bool = True,
        # RoPE params
        rope_theta: float = 500000.0,
        linear_scaling: typing.Optional[typing.Dict] = None,
        # FFN params
        ffn_activation: str = "silu",
        ffn_dim_multiplier: typing.Optional[float] = None,
        multiple_of: int = 256,
        # Output head params
        lm_head_bias: bool = False,
    ):
        """Initialize Llama 3 model."""
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.block_size = block_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # Token embeddings (no position embeddings in Llama)
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
        )

        # RoPE module - computed once and shared across all blocks
        self.rope = rope.PrecomputedRoPE(
            dim=hidden_dim // num_heads,  # head_dim
            theta=rope_theta,
            linear_scaling=linear_scaling,
        )

        # Transformer blocks
        self.blocks: nn.ModuleList[llama3_blocks.Llama3TransformerBlock] = nn.ModuleList(
            [
                llama3_blocks.Llama3TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    rope_module=self.rope,
                    dropout=dropout,
                    norm_eps=norm_eps,
                    ffn_activation=ffn_activation,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    multiple_of=multiple_of,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(n_layers)
            ]
        )

        # Final RMSNorm
        self.ln_f = nn.RMSNorm(hidden_dim, eps=norm_eps)

        # Output projection - separate (not tied like GPT-2)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=lm_head_bias)

        # Note: PyTorch's default initialization is fine for Llama3
        # No special weight initialization needed

    @classmethod
    def from_config(cls, config: llama.Llama3Config) -> "Llama3":
        """Create Llama3 from a config object."""
        return cls(
            # Core dimensions
            vocab_size=config.vocab_size,
            hidden_dim=config.transformer.hidden_dim,
            n_layers=config.transformer.n_layers,
            block_size=config.transformer.block_size,
            # Transformer params
            num_heads=config.transformer.attention.num_heads,
            num_kv_heads=config.transformer.attention.num_kv_heads,
            dropout=config.transformer.dropout,
            norm_eps=config.transformer.normalization.norm_eps,
            use_flash_attention=config.transformer.attention.use_flash_attention,
            # RoPE params
            rope_theta=config.transformer.rope.theta,
            linear_scaling=config.transformer.rope.linear_scaling,
            # FFN params
            ffn_activation=config.transformer.ffn.activation,
            ffn_dim_multiplier=config.transformer.ffn.ffn_dim_multiplier,
            multiple_of=config.transformer.ffn.multiple_of
            if config.transformer.ffn.multiple_of
            else 256,
            # Output head params
            lm_head_bias=config.output_head.lm_head_bias,
        )

    def _compute_logits(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq vocab"]:
        """
        Compute output logits using separate lm_head.

        Llama 3 uses a separate output projection (not tied embeddings like GPT-2).
        This allows the model to learn different representations for input and output.
        """
        return self.lm_head(hidden_states)

    def forward(
        self,
        input_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> outputs.ForwardOutput:
        """
        Process input tokens through the Llama 3 model.

        The Llama 3 forward pass:
        1. Token embeddings - convert token IDs to dense vectors
        2. Transformer blocks - apply N layers of GQA (with RoPE) and SwiGLU FFN
        3. Final RMSNorm - normalize the final hidden states
        4. Output projection - project to vocabulary size using separate lm_head

        Note: No position embeddings are added to input; RoPE is applied during attention.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask for padding
            position_offset: Position offset for RoPE (used during generation with KV cache)
        """
        # Token embeddings only (no position embeddings)
        x = self.token_embeddings(input_ids)

        # Apply transformer blocks with RoPE handled internally
        hidden_states, _ = self._apply_transformer_blocks(
            x,
            blocks=self.blocks,
            attention_mask=attention_mask,
            position_offset=position_offset,
        )

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Output projection
        logits = self._compute_logits(hidden_states)

        return outputs.ForwardOutput(logits=logits)

    def generate(
        self,
        token_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: typing.Optional[int] = None,
        use_cache: bool = True,
    ) -> jaxtyping.Int[torch.Tensor, "batch new_seq"]:
        """
        Generate tokens using cached or standard generation.

        Supports KV caching for efficient generation when use_cache=True.
        """
        if use_cache:
            # Use cached generation for efficiency
            return self.generate_with_cache(
                model_forward=self.forward,
                setup_cache_fn=self.setup_kv_cache,
                reset_cache_fn=self.reset_kv_cache,
                block_size=self.block_size,
                token_ids=token_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        else:
            raise NotImplementedError(
                "Non-cached generation not implemented for Llama3. "
                "Set use_cache=True or add AutoregressiveGenerationMixin to the class."
            )

    def get_fsdp_wrappable_modules(self) -> typing.Set[typing.Type[nn.Module]]:
        """Return module types that FSDP should wrap."""
        return {llama3_blocks.Llama3TransformerBlock}

    def get_transformer_blocks(self) -> typing.List[nn.Module]:
        """Return transformer block instances."""
        return list(self.blocks)

    def setup_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        device: typing.Union[torch.device, str] = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Setup KV cache for generation."""
        # Setup cache in each attention layer
        for block in self.blocks:
            block.attention.setup_cache(
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                n_kv_heads=self.num_kv_heads,
                head_dim=self.hidden_dim // self.num_heads,
                dtype=dtype,
                device=device,
            )

    def reset_kv_cache(self) -> None:
        """Reset KV cache after generation."""
        # Reset cache in each attention layer
        for block in self.blocks:
            block.attention.reset_cache()

    def get_fsdp_special_modules(self) -> typing.Dict[str, nn.Module]:
        """Return special modules for FSDP wrapping."""
        return super().get_fsdp_special_modules(
            token_embeddings=self.token_embeddings,
            lm_head=self.lm_head,
        )
