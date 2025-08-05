# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from pretraining.common.base import llm
from pretraining.common.base import outputs
from pretraining.common.patterns.blocks import llama3 as llama3_blocks
from pretraining.common.patterns.cache import kv_cache
from pretraining.common.patterns.position import core
from pretraining.configs.model.architectures import llama


class Llama3(llm.BaseLLM):
    """
    Llama Language Model implementation.

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
        num_kv_heads: int,
        # RoPE params
        rope_theta: float = 500000.0,
        norm_eps: float = 1e-5,
        use_flash_attention: bool = False,
        # FFN params
        ffn_dim_multiplier: float = 1.3,
        multiple_of: int = 256,
        # Output head params
        lm_head_bias: bool = False,
    ):
        """
        Initialize Llama3 model.

        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension
            n_layers: Number of transformer layers
            block_size: Maximum sequence length
            rope_theta: Base frequency for RoPE (Llama3 default: 500000)
            num_heads: Number of attention heads
            num_kv_heads: Number of key/value heads (for GQA)
            norm_eps: RMSNorm epsilon
            use_flash_attention: Whether to use flash attention
            ffn_dim_multiplier: Multiplier for FFN dimension
            multiple_of: Round FFN dimension to multiple of this
            lm_head_bias: Whether to use bias in lm_head (traditionally False)
        """
        super().__init__()

        # Store dimensions
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.block_size = block_size

        # Token embeddings
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
        )

        # RoPE module - standard Llama3 configuration without scaling
        self.rope = core.PrecomputedRoPE(
            dim=hidden_dim // num_heads,  # head_dim
            theta=rope_theta,
        )

        # Transformer blocks
        self.blocks: nn.ModuleList[llama3_blocks.Llama3TransformerBlock] = nn.ModuleList(
            [
                llama3_blocks.Llama3TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    rope_module=self.rope,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    multiple_of=multiple_of,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(n_layers)
            ]
        )

        # Final RMSNorm
        self.norm = nn.RMSNorm(hidden_dim, eps=norm_eps)

        # Output projection - separate (not tied)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=lm_head_bias)

    @classmethod
    def from_config(cls, config: llama.Llama3Config) -> "Llama3":
        """Create Llama3 from a config object."""
        attn_config = config.transformer.attention

        return cls(
            # Core dimensions
            vocab_size=config.token_embedding.vocab_size,
            hidden_dim=config.transformer.hidden_dim,
            n_layers=config.transformer.n_layers,
            block_size=config.transformer.block_size,
            # RoPE params
            rope_theta=config.transformer.rope.theta,
            # Transformer params
            num_heads=attn_config.num_heads,
            num_kv_heads=attn_config.num_kv_heads,
            norm_eps=config.transformer.normalization.norm_eps,
            use_flash_attention=attn_config.use_flash_attention,
            # FFN params
            ffn_dim_multiplier=config.transformer.ffn.ffn_dim_multiplier,
            multiple_of=config.transformer.ffn.multiple_of,
            # Output head params
            lm_head_bias=config.output_head.lm_head_bias,
        )

    def _apply_transformer_blocks(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
        output_hidden_states: bool = False,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        typing.Optional[typing.List[jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]]],
    ]:
        """Pass input through all transformer blocks."""
        # Store hidden states if requested
        all_hidden_states = [] if output_hidden_states else None

        # Process through each block
        hidden: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"] = x
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(hidden)

            # Forward through block
            hidden = block(
                hidden,
                attention_mask=attention_mask,
                position_offset=position_offset,
            )

        # Apply final norm
        normalized: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        normalized = self.norm(hidden)

        if output_hidden_states:
            all_hidden_states.append(normalized)

        return normalized, all_hidden_states

    def _compute_logits(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len vocab"]:
        """Compute output logits from hidden states."""
        # Llama uses separate output projection
        logits: jaxtyping.Float[torch.Tensor, "batch seq_len vocab"]
        logits = self.lm_head(hidden_states)
        return logits

    def forward(
        self,
        input_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> outputs.ForwardOutput:
        """
        Unified forward pass for both training and inference.

        Args:
            input_ids: Input token indices [batch, seq]
            attention_mask: Optional attention mask
            position_offset: Starting position for RoPE (for KV cache during generation)

        Returns:
            ForwardOutput containing logits
        """
        # 1. Token embeddings
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        x = self.token_embeddings(input_ids)

        # 2. Apply transformer blocks with position offset
        hidden_states, _ = self._apply_transformer_blocks(
            x,
            attention_mask=attention_mask,
            position_offset=position_offset,
            output_hidden_states=False,
        )

        # 3. Compute logits
        logits = self._compute_logits(hidden_states)

        # 4. Return ForwardOutput
        return outputs.ForwardOutput(logits=logits)

    def install_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        dtype: torch.dtype = torch.float16,
        device: typing.Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Install static KV cache in all attention layers for efficient generation.

        Args:
            batch_size: Maximum batch size to support
            max_seq_length: Maximum sequence length to cache
            dtype: Data type for cache tensors
            device: Device to allocate cache on
        """
        for block in self.blocks:
            block.attention.cache = kv_cache.KVCache(
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                n_kv_heads=block.attention.num_kv_heads,
                head_dim=block.attention.head_dim,
                dtype=dtype,
                device=device,
            )

    def clear_kv_cache(self) -> None:
        """Remove KV cache from all attention layers."""
        for block in self.blocks:
            if hasattr(block.attention, "cache"):
                block.attention.cache = None

    @torch.no_grad()
    def generate(
        self,
        idx: jaxtyping.Int[torch.Tensor, "batch seq"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: typing.Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> jaxtyping.Int[torch.Tensor, "batch total_seq"]:
        """
        Generate tokens autoregressively.

        Args:
            idx: Initial prompt tokens [batch, seq]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            use_cache: Whether to use KV cache (recommended for efficiency)
            **kwargs: Additional generation parameters

        Returns:
            Generated token sequence including prompt
        """
        device = idx.device
        batch_size = idx.shape[0]
        seq_len = idx.shape[1]

        # Install KV cache if requested
        if use_cache:
            total_len = seq_len + max_new_tokens
            self.install_kv_cache(
                batch_size=batch_size,
                max_seq_length=total_len,
                dtype=self.token_embeddings.weight.dtype,
                device=device,
            )

        # Process initial prompt
        position_offset = 0
        if use_cache:
            # Process full prompt with position offset 0
            model_output = self.forward(
                input_ids=idx,
                attention_mask=kwargs.get("attention_mask"),
                position_offset=position_offset,
            )
            logits = model_output.logits
            position_offset = seq_len
        else:
            # Without cache, we'll process the full sequence each time
            model_output = self.forward(
                input_ids=idx,
                attention_mask=kwargs.get("attention_mask"),
                position_offset=0,
            )
            logits = model_output.logits

        # Start generation loop
        generated = idx
        for _ in range(max_new_tokens):
            # Get logits for last position
            logits_last = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                logits_last = logits_last / temperature

            # Apply top-k if specified
            if top_k is not None:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = -float("Inf")

            # Sample from distribution
            probs = torch.nn.functional.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat((generated, idx_next), dim=1)

            # Get logits for next token
            if use_cache:
                # Only process the new token with updated position
                model_output = self.forward(
                    input_ids=idx_next,
                    attention_mask=None,
                    position_offset=position_offset,
                )
                logits = model_output.logits
                position_offset += 1
            else:
                # Process full sequence (less efficient)
                # Crop if sequence is too long
                if generated.size(1) > self.block_size:
                    generated = generated[:, -self.block_size :]
                model_output = self.forward(
                    input_ids=generated,
                    attention_mask=kwargs.get("attention_mask"),
                    position_offset=0,
                )
                logits = model_output.logits

        # Clean up KV cache
        if use_cache:
            self.clear_kv_cache()

        return generated
