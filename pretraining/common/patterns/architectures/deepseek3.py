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
from pretraining.configs.model.components import position


class DeepSeek3(llm.BaseLLM):
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

    def __init__(
        self,
        # Core dimensions
        vocab_size: int,
        hidden_dim: int,
        n_layers: int,
        block_size: int,
        # Token embedding params
        embedding_dropout: float = 0.0,
        # RoPE params
        rope_theta: float = 10000.0,
        rope_dim: int = 64,  # Partial RoPE dimension for MLA
        # MLA attention params
        num_heads: int = 128,
        head_dim: int = 128,
        kv_compression_dim: int = 512,
        query_compression_dim: int = 1536,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        use_flash_attention: bool = False,
        # MoE params
        num_experts: int = 256,
        num_experts_per_token: int = 6,
        # Output head params
        lm_head_bias: bool = False,
        # Multi-token prediction params (required for DeepSeek)
        mtp_n_predict: int = 3,
        mtp_dropout_rate: float = 0.1,
    ):
        """
        Initialize DeepSeek3 model.

        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension (d_model)
            n_layers: Number of transformer layers
            block_size: Maximum sequence length
            embedding_dropout: Dropout for embeddings
            rope_theta: Base frequency for RoPE
            rope_dim: Dimension for partial RoPE (MLA specific)
            num_heads: Number of attention heads
            head_dim: Dimension per head
            kv_compression_dim: Compression dimension for K/V
            query_compression_dim: Compression dimension for Q
            dropout: Dropout rate throughout model
            norm_eps: RMSNorm epsilon
            use_flash_attention: Whether to use flash attention
            num_experts: Number of MoE experts
            num_experts_per_token: How many experts each token uses
            lm_head_bias: Whether to use bias in lm_head
            mtp_n_predict: Number of future tokens to predict
            mtp_dropout_rate: Dropout rate for MTP heads
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

        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(embedding_dropout)

        # RoPE module - Partial RoPE for MLA
        rope_config = position.RoPEConfig(theta=rope_theta)
        self.rope = rope_partial.PartialRoPE(
            dim=rope_dim,
            config=rope_config,
        )

        # Transformer blocks
        self.blocks: nn.ModuleList[deepseek3_blocks.DeepSeek3TransformerBlock] = nn.ModuleList(
            [
                deepseek3_blocks.DeepSeek3TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    kv_compression_dim=kv_compression_dim,
                    query_compression_dim=query_compression_dim,
                    rope_module=self.rope,
                    rope_dim=rope_dim,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    dropout=dropout,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
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
            dropout_rate=mtp_dropout_rate,
        )

    @classmethod
    def from_config(cls, config: deepseek.DeepSeek3Config) -> "DeepSeek3":
        """Create DeepSeek3 from a config object."""
        # Extract MLA attention config
        attn_config = config.transformer.attention

        # Extract MoE config
        moe_config = config.transformer.moe

        return cls(
            # Core dimensions
            vocab_size=config.token_embedding.vocab_size,
            hidden_dim=config.transformer.hidden_dim,
            n_layers=config.transformer.n_layers,
            block_size=config.transformer.block_size,
            # Token embedding params
            embedding_dropout=config.token_embedding.embedding_dropout,
            # RoPE params
            rope_theta=config.transformer.rope.theta,
            rope_dim=attn_config.rope_dim,
            # MLA attention params
            num_heads=attn_config.num_heads,
            head_dim=attn_config.head_dim,
            kv_compression_dim=attn_config.kv_compression_dim,
            query_compression_dim=attn_config.query_compression_dim,
            dropout=config.transformer.dropout,
            norm_eps=config.transformer.normalization.norm_eps,
            use_flash_attention=attn_config.use_flash_attention,
            # MoE params
            num_experts=moe_config.num_experts,
            num_experts_per_token=moe_config.num_experts_per_token,
            # Output head params
            lm_head_bias=config.output_head.lm_head_bias,
            # MTP params (required)
            mtp_n_predict=config.mtp.n_predict,
            mtp_dropout_rate=config.mtp.dropout_rate,
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
