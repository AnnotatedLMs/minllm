# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from pretraining.common.base import llm
from pretraining.common.patterns.attention import grouped_query
from pretraining.common.patterns.blocks import llama3 as llama3_blocks
from pretraining.common.patterns.cache import kv_cache
from pretraining.common.patterns.position import rope
from pretraining.configs.model.architectures import llama


class LlamaLLM(llm.BaseLLM):
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

    def __init__(self, config: llama.Llama3Config):
        super().__init__()
        self.config = config

        # Validate this is a Llama config
        # Llama doesn't have position_embedding field at all
        assert config.transformer.rope is not None, "Llama requires RoPE config"

        # Extract dimensions
        self.vocab_size = config.token_embedding.vocab_size
        self.hidden_dim = config.transformer.hidden_dim
        self.n_layers = config.transformer.n_layers
        self.block_size = config.transformer.block_size

        # Token embeddings
        self.token_embeddings = nn.Embedding(
            num_embeddings=config.token_embedding.vocab_size,
            embedding_dim=config.token_embedding.embedding_dim,
        )

        # No position embeddings for Llama (uses RoPE instead)

        # Dropout for embeddings (if specified)
        self.embed_dropout = nn.Dropout(config.token_embedding.embedding_dropout)

        # RoPE module (shared across all layers)
        rope_config = config.transformer.rope
        self.rope = rope.RoPE(
            dim=self.hidden_dim // config.transformer.attention.num_heads,  # head_dim
            config=rope_config,
        )

        # Transformer blocks
        self.blocks: nn.ModuleList[llama3_blocks.Llama3TransformerBlock] = nn.ModuleList(
            [self._create_transformer_block(config) for _ in range(config.transformer.n_layers)]
        )

        # Final RMSNorm
        self.norm = nn.RMSNorm(
            self.hidden_dim,
            eps=config.transformer.normalization.norm_eps,
        )

        # Output projection (language modeling head)
        # Llama does NOT use tied embeddings - has separate output projection
        self.lm_head = nn.Linear(
            self.hidden_dim, self.vocab_size, bias=config.output_head.lm_head_bias
        )

        # Apply weight initialization from config
        self._apply_weight_initialization(config)

    def _create_transformer_block(
        self, config: llama.Llama3Config
    ) -> llama3_blocks.Llama3TransformerBlock:
        """Create a Llama style transformer block."""
        attn_config = config.transformer.attention
        ffn_config = config.transformer.ffn
        norm_config = config.transformer.normalization

        # Extract GQA parameters
        num_kv_heads = None
        if hasattr(attn_config, "num_kv_heads"):
            num_kv_heads = attn_config.num_kv_heads

        # Llama uses Llama3TransformerBlock with SwiGLU and GQA
        return llama3_blocks.Llama3TransformerBlock(
            hidden_dim=self.hidden_dim,
            num_heads=attn_config.num_heads,
            num_kv_heads=num_kv_heads if num_kv_heads else attn_config.num_heads,
            rope_module=self.rope,
            dropout=config.transformer.dropout,
            norm_eps=norm_config.norm_eps,
            ffn_dim_multiplier=ffn_config.ffn_dim_multiplier,
            multiple_of=ffn_config.multiple_of,
            use_flash_attention=attn_config.use_flash_attention,
        )

    def _apply_transformer_blocks(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
        output_hidden_states: bool = False,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
        typing.Optional[typing.List[jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]]],
    ]:
        """Pass input through all transformer blocks."""
        # Store hidden states if requested
        all_hidden_states = [] if output_hidden_states else None

        # Process through each block
        hidden: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"] = x
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(hidden)

            # Note: KV caching is now handled internally by attention layers

            # Forward through block
            hidden = block(
                hidden,
                attention_mask=attention_mask,
                position_offset=position_offset,
            )

            # Note: KV caching is now handled internally by attention layers

        # Apply final norm
        normalized: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]
        normalized = self.norm(hidden)

        if output_hidden_states:
            all_hidden_states.append(normalized)

        return normalized, all_hidden_states

    def _compute_logits(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq vocab"]:
        """Compute output logits from hidden states."""
        # Llama uses separate output projection
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"]
        logits = self.lm_head(hidden_states)
        return logits

    def training_forward(
        self,
        idx: jaxtyping.Int[torch.Tensor, "batch seq"],
        targets: jaxtyping.Int[torch.Tensor, "batch seq"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> typing.Union[
        torch.Tensor,
        typing.Tuple[torch.Tensor, jaxtyping.Float[torch.Tensor, "batch seq vocab"]],
        typing.Tuple[
            torch.Tensor,
            jaxtyping.Float[torch.Tensor, "batch seq vocab"],
            typing.Dict[str, typing.Any],
        ],
    ]:
        """
        Forward pass for training - no KV cache.

        Args:
            idx: Input token indices [batch, seq]
            targets: Target tokens for loss computation
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states
            **kwargs: Additional arguments (for compatibility)

        Returns:
            - Just loss by default
            - (loss, logits) if return_logits=True in kwargs
            - (loss, logits, extras) if output_hidden_states=True
        """
        # 1. Embed tokens and apply dropout
        x: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]
        x = self.token_embeddings(idx)
        x = self.embed_dropout(x)

        # 2. Apply transformer blocks (no position offset for training)
        hidden_states, all_hidden_states = self._apply_transformer_blocks(
            x,
            attention_mask=attention_mask,
            position_offset=0,  # Always 0 for training
            output_hidden_states=output_hidden_states,
        )

        # 3. Compute logits
        logits = self._compute_logits(hidden_states)

        # 4. Compute loss
        loss = self.compute_loss(logits, targets)

        # Return based on what's requested
        return_logits = kwargs.get("return_logits", False)
        if output_hidden_states:
            extras = {"hidden_states": all_hidden_states}
            return loss, logits, extras
        elif return_logits:
            return loss, logits
        else:
            return loss

    @torch.no_grad()
    def inference_forward(
        self,
        idx: jaxtyping.Int[torch.Tensor, "batch seq"],
        position_offset: int = 0,
        attention_mask: typing.Optional[torch.Tensor] = None,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq vocab"]:
        """
        Forward pass for inference/generation with KV cache support.

        This method assumes install_kv_cache() has been called first.

        Args:
            idx: Input token indices [batch, seq]
            position_offset: Starting position for RoPE (for incremental generation)
            attention_mask: Optional attention mask
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Logits tensor [batch, seq, vocab]
        """
        # 1. Embed tokens (no dropout during inference)
        x: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]
        x = self.token_embeddings(idx)

        # 2. Apply transformer blocks with position offset
        hidden_states, _ = self._apply_transformer_blocks(
            x,
            attention_mask=attention_mask,
            position_offset=position_offset,
            output_hidden_states=False,
        )

        # 3. Compute logits
        logits = self._compute_logits(hidden_states)

        return logits

    def compute_loss(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
        targets: jaxtyping.Int[torch.Tensor, "batch seq"],
        ignore_index: int = -100,
        **kwargs,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for language modeling."""
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute loss
        loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)

        return loss

    def get_num_params(self) -> int:
        """
        Get number of parameters in the model.

        Returns:
            Total number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

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
            if isinstance(block.attention, grouped_query.GroupedQueryAttention):
                # Get number of KV heads from attention layer
                n_kv_heads = block.attention.num_kv_heads
                head_dim = block.attention.head_dim

                # Create and install cache
                block.attention.cache = kv_cache.KVCache(
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
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
            logits = self.inference_forward(idx, position_offset=position_offset)
            position_offset = seq_len
        else:
            # Without cache, we'll process the full sequence each time
            logits = self.inference_forward(idx)

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
                logits = self.inference_forward(idx_next, position_offset=position_offset)
                position_offset += 1
            else:
                # Process full sequence (less efficient)
                # Crop if sequence is too long
                if generated.size(1) > self.block_size:
                    generated = generated[:, -self.block_size :]
                logits = self.inference_forward(generated)

        # Clean up KV cache
        if use_cache:
            self.clear_kv_cache()

        return generated
