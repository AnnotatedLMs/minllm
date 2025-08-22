# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.mixins import fsdp_mixins
from pretraining.common.mixins import generation_mixins
from pretraining.common.models import outputs
from pretraining.common.models.blocks import llama3_blocks
from pretraining.common.models.position import rope
from pretraining.configs.model.architectures import llama


class Llama3(
    nn.Module,
    fsdp_mixins.FSDPMixin,
    generation_mixins.CachedGenerationMixin,
):
    """
    Llama 3 language model - Efficient architecture with GQA and RoPE.
    Llama 3 Team, 2024, https://arxiv.org/pdf/2407.21783

    Step-by-step control flow (model-level orchestration):
    1. Token embedding: Convert token IDs to dense vectors
    2. Transformer blocks: N layers of Llama3TransformerBlock
       - Each block: RMSNorm → GQA (with RoPE) → RMSNorm → SwiGLU
    3. Final RMSNorm: Normalize final hidden states
    4. Output projection: Separate lm_head (no weight tying)

    Learning components (what learns across the model):
    - Token embeddings: Learn semantic representations for each token
    - PrecomputedRoPE: NO learning - fixed sinusoidal rotations
    - Llama3TransformerBlocks: Each contains learnable GQA and SwiGLU
    - Final RMSNorm: Learns single scale parameter (no bias)
    - Output projection: Separate linear layer learns output mappings

    Key architectural choices:
    - GQA with 8 KV heads: 4x memory reduction during inference
    - RoPE with θ=500,000: Supports 32K+ context without retraining
    - SwiGLU activation: Better performance through gating mechanism
    - No biases: Cleaner optimization, ~10M fewer parameters
    - RMSNorm: Simpler than LayerNorm, equally effective
    - Separate lm_head: Better scaling than weight tying

    Mixin contributions:
    - TransformerBlockStackMixin: Applies blocks with position offset support
    - CachedGenerationMixin: Efficient KV-cached autoregressive generation
    - FSDPMixin: Enables efficient distributed training
    - No weight init mixin: Pytorch defaults work well for this architecture

    Memory optimizations:
    - GQA reduces KV cache from O(n·h·d) to O(n·h/g·d) where g=4
    - PrecomputedRoPE: O(1) position encoding via precomputed tables
    - Flash attention compatible for additional memory savings
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

        self.ln_f = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=lm_head_bias)

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
        x = self.token_embeddings(input_ids)

        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"] = x
        for block in self.blocks:
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, position_offset=position_offset
            )

        hidden_states = self.ln_f(hidden_states)

        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"]
        logits = self.lm_head(hidden_states)

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
