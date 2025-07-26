# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from pretraining.common.base import llm
from pretraining.common.patterns.blocks import deepseek3 as deepseek3_blocks
from pretraining.common.patterns.heads import multi_token
from pretraining.common.patterns.position import rope_partial
from pretraining.configs.model.architectures import deepseek


class DeepSeekLLM(llm.BaseLLM):
    """
    DeepSeek Language Model implementation.

    Used by: DeepSeek-V3

    Variation: Multi-head Latent Attention (MLA) with PartialRoPE
    Computation: Compresses Q/K/V through bottleneck, applies RoPE to position-only features
    Effect: Reduces activation memory while separating content and position processing

    Variation: Auxiliary-loss-free MoE instead of FFN
    Computation: Shared expert + routed experts with dynamic bias adjustment
    Effect: Automatically balances expert usage without explicit load balancing loss - the gate
            biases update based on how many tokens each expert processes, steering future
            tokens away from overloaded experts

    Variation: Optional multi-token prediction heads
    Computation: Additional heads predict n+1, n+2, ... tokens
    Effect: Improves sample efficiency during training
    """

    def __init__(self, config: deepseek.DeepSeek3Config):
        super().__init__()
        self.config = config

        # DeepSeek doesn't have position_embedding field at all
        assert config.transformer.rope is not None, "DeepSeek requires RoPE config"
        assert config.transformer.moe is not None, "DeepSeek requires MoE config"

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

        # No position embeddings for DeepSeek (uses RoPE instead)

        # Dropout for embeddings (if specified)
        self.embed_dropout = nn.Dropout(config.token_embedding.embedding_dropout)

        # RoPE module (shared across all layers)
        rope_config = config.transformer.rope
        # For MLA, we need to use the rope_dim from attention config
        attn_config = config.transformer.attention
        rope_dim = attn_config.rope_dim  # Required field for DeepSeek MLA

        # DeepSeek uses PartialRoPE for position-only vectors
        self.rope = rope_partial.PartialRoPE(
            dim=rope_dim,
            config=rope_config,
        )

        # Transformer blocks
        self.blocks: nn.ModuleList[deepseek3_blocks.DeepSeek3TransformerBlock] = nn.ModuleList(
            [self._create_transformer_block(config) for _ in range(config.transformer.n_layers)]
        )

        # Final RMSNorm
        self.norm = nn.RMSNorm(
            self.hidden_dim,
            eps=config.transformer.normalization.norm_eps,
        )

        # Output projection (language modeling head)
        # DeepSeek does NOT use tied embeddings - has separate output projection
        self.lm_head = nn.Linear(
            self.hidden_dim, self.vocab_size, bias=config.output_head.lm_head_bias
        )

        # Multi-token prediction heads (optional)
        self.mtp_heads = None
        if config.mtp is not None:
            self.mtp_heads = multi_token.MultiTokenPredictionHead(
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size,
                depth=config.mtp.n_predict,
                dropout_rate=config.mtp.dropout_rate,
            )

        # Apply weight initialization from config
        self._apply_weight_initialization(config)

    def _create_transformer_block(
        self, config: deepseek.DeepSeek3Config
    ) -> deepseek3_blocks.DeepSeek3TransformerBlock:
        """Create a DeepSeek style transformer block."""
        attn_config = config.transformer.attention
        moe_config = config.transformer.moe
        norm_config = config.transformer.normalization

        # DeepSeek uses DeepSeek3TransformerBlock with MoE and MLA
        return deepseek3_blocks.DeepSeek3TransformerBlock.from_mla_config(
            hidden_dim=self.hidden_dim,
            mla_config=attn_config,
            rope_module=self.rope,
            num_experts=moe_config.num_experts,
            num_experts_per_token=moe_config.num_experts_per_token,
            dropout=config.transformer.dropout,
            norm_eps=norm_config.norm_eps,
            use_flash_attention=attn_config.use_flash_attention,
        )

    def _apply_transformer_blocks(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
        typing.Optional[typing.List[jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]]],
        typing.List[torch.Tensor],  # Auxiliary losses
    ]:
        """Pass input through all transformer blocks."""
        # Store hidden states if requested
        all_hidden_states = [] if output_hidden_states else None

        # Store auxiliary losses from MoE layers
        aux_losses = []

        # Process through each block
        hidden: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"] = x
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(hidden)

            # Forward through block (no KV cache for DeepSeek)
            hidden = block(
                hidden,
                attention_mask=attention_mask,
            )

            # Collect auxiliary loss from MoE if present
            if (
                hasattr(block, "moe")
                and hasattr(block.moe, "aux_loss")
                and block.moe.aux_loss is not None
            ):
                aux_losses.append(block.moe.aux_loss)

        # Apply final norm
        normalized: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]
        normalized = self.norm(hidden)

        if output_hidden_states:
            all_hidden_states.append(normalized)

        return normalized, all_hidden_states, aux_losses

    def _compute_logits(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq vocab"]:
        """Compute output logits from hidden states."""
        # DeepSeek uses separate output projection
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"]
        logits = self.lm_head(hidden_states)
        return logits

    def _compute_mtp_logits(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
    ) -> typing.Optional[typing.List[jaxtyping.Float[torch.Tensor, "batch seq vocab"]]]:
        """Compute multi-token prediction logits if enabled."""
        if self.mtp_heads is not None:
            return self.mtp_heads(hidden_states)
        return None

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
        Forward pass for training.

        Args:
            idx: Input token indices [batch, seq]
            targets: Target tokens for loss computation
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states
            **kwargs: Additional arguments (for compatibility)

        Returns:
            - Just loss by default
            - (loss, logits) if return_logits=True in kwargs
            - (loss, logits, extras) if additional outputs requested
        """
        # 1. Embed tokens and apply dropout
        x: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]
        x = self.token_embeddings(idx)
        x = self.embed_dropout(x)

        # 2. Apply transformer blocks
        hidden_states, all_hidden_states, aux_losses = self._apply_transformer_blocks(
            x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        # 3. Compute logits
        logits = self._compute_logits(hidden_states)

        # 4. Compute MTP logits if enabled
        mtp_logits = self._compute_mtp_logits(hidden_states)

        # 5. Compute loss
        loss = self.compute_loss(logits, targets, aux_losses=aux_losses)

        # Return based on what's requested
        return_logits = kwargs.get("return_logits", False)
        if output_hidden_states or mtp_logits is not None or aux_losses:
            extras = {}
            if output_hidden_states:
                extras["hidden_states"] = all_hidden_states
            if mtp_logits is not None:
                extras["mtp_logits"] = mtp_logits
            if aux_losses:
                extras["aux_losses"] = aux_losses
            return loss, logits, extras
        elif return_logits:
            return loss, logits
        else:
            return loss

    @torch.no_grad()
    def inference_forward(
        self,
        idx: jaxtyping.Int[torch.Tensor, "batch seq"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq vocab"]:
        """
        Forward pass for inference/generation.

        Note: DeepSeek doesn't use KV cache.

        Args:
            idx: Input token indices [batch, seq]
            attention_mask: Optional attention mask
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Logits tensor [batch, seq, vocab]
        """
        # 1. Embed tokens (no dropout during inference)
        x: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]
        x = self.token_embeddings(idx)

        # 2. Apply transformer blocks
        hidden_states, _, _ = self._apply_transformer_blocks(
            x,
            attention_mask=attention_mask,
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
        aux_losses: typing.Optional[typing.List[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for language modeling with auxiliary losses."""
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute main loss
        main_loss = nn.functional.cross_entropy(
            shift_logits, shift_labels, ignore_index=ignore_index
        )

        # Add auxiliary losses from MoE
        total_loss = main_loss
        if aux_losses:
            aux_loss = sum(aux_losses) / len(aux_losses)
            total_loss = main_loss + aux_loss

        return total_loss

    def get_num_params(self) -> int:
        """
        Get number of parameters in the model.

        Returns:
            Total number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
