# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.base import llm
from pretraining.common.base import outputs
from pretraining.common.patterns.blocks import deepseek3 as deepseek3_blocks
from pretraining.common.patterns.heads import multi_token
from pretraining.common.patterns.position import rope_partial
from pretraining.configs.model.architectures import deepseek
from pretraining.utils.training import fsdp_policies


class DeepSeek3(llm.BaseLLM, fsdp_policies.FSDPMixin):
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
    ):
        """
        Initialize DeepSeek3 model.

        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension (hidden_dim)
            n_layers: Number of transformer layers
            block_size: Maximum sequence length
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
        self.embedding_dropout = nn.Dropout(dropout) if dropout is not None else None

        # RoPE module - Partial RoPE for MLA
        self.rope = rope_partial.PartialRoPE(
            dim=rope_dim,
            theta=rope_theta,
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
            dropout=dropout,
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
        )

    def _apply_transformer_blocks(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        typing.Optional[typing.List[jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]]],
        typing.List[torch.Tensor],  # Auxiliary losses
    ]:
        """Pass input through all transformer blocks."""
        # Store hidden states if requested
        all_hidden_states = [] if output_hidden_states else None

        # Store auxiliary losses from MoE layers
        aux_losses = []

        # Process through each block
        hidden: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"] = x
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(hidden)

            # Forward through block (no KV cache for DeepSeek)
            hidden = block(
                hidden,
                attention_mask=attention_mask,
            )

            # Collect auxiliary loss from MoE
            aux_loss = block.moe.get_auxiliary_loss()
            if aux_loss is not None:
                aux_losses.append(aux_loss)

        # Apply final norm
        normalized: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        normalized = self.norm(hidden)

        if output_hidden_states:
            all_hidden_states.append(normalized)

        return normalized, all_hidden_states, aux_losses

    def _compute_logits(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len vocab"]:
        """Compute output logits from hidden states."""
        # DeepSeek uses separate output projection
        logits: jaxtyping.Float[torch.Tensor, "batch seq_len vocab"]
        logits = self.lm_head(hidden_states)
        return logits

    def _compute_mtp_logits(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> typing.List[jaxtyping.Float[torch.Tensor, "batch seq_len vocab"]]:
        """Compute multi-token prediction logits."""
        return self.mtp_heads(hidden_states)

    def forward(
        self,
        input_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> outputs.ForwardOutput:
        """
        Unified forward pass for both training and inference.

        Args:
            input_ids: Input token indices [batch, seq]
            attention_mask: Optional attention mask

        Returns:
            ForwardOutput containing:
                - logits: Main output logits
                - mtp_logits: MTP logits (only during training)
                - aux_losses: MoE auxiliary losses (only during training)
        """
        # 1. Token embeddings with dropout
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        x = self.token_embeddings(input_ids)
        # Note: PyTorch's Train/Eval Mode automatically handles dropout
        x = self._maybe_apply_dropout(x, self.embedding_dropout)

        # 2. Apply transformer blocks
        aux_losses = []
        hidden_states = x
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

            # Collect auxiliary losses only during training
            if self.training:
                aux_loss = block.moe.get_auxiliary_loss()
                if aux_loss is not None:
                    aux_losses.append(aux_loss)

        # 3. Final layer norm
        hidden_states = self.norm(hidden_states)

        # 4. Compute main logits
        logits = self._compute_logits(hidden_states)

        # 5. Compute MTP logits only during training
        mtp_logits = None
        if self.training:
            mtp_logits = self._compute_mtp_logits(hidden_states)

        # 6. Return ForwardOutput
        return outputs.ForwardOutput(
            logits=logits, mtp_logits=mtp_logits, aux_losses=aux_losses if aux_losses else None
        )

    @torch.no_grad()
    def generate(
        self,
        token_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: typing.Optional[int] = None,
        **kwargs,
    ) -> jaxtyping.Int[torch.Tensor, "batch new_seq"]:
        """
        Generate tokens autoregressively with DeepSeek3.

        Note: This implementation uses the main prediction head only.
        Future versions could leverage MTP predictions for improved generation.

        Args:
            token_ids: Initial token indices [batch, seq]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k most likely tokens
            **kwargs: Additional generation arguments

        Returns:
            Generated token indices [batch, seq + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            context_tokens = (
                token_ids
                if token_ids.size(1) <= self.block_size
                else token_ids[:, -self.block_size :]
            )

            # Forward the model to get logits
            model_output = self.forward(
                input_ids=context_tokens,
                attention_mask=kwargs.get("attention_mask"),
            )
            logits = model_output.logits

            # Focus on the last token logits
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            token_ids = torch.cat((token_ids, next_token), dim=1)

        return token_ids

    def get_fsdp_wrappable_modules(self) -> typing.Set[typing.Type[nn.Module]]:
        return {deepseek3_blocks.DeepSeek3TransformerBlock}

    def get_transformer_blocks(self) -> typing.List[nn.Module]:
        return list(self.blocks)

    def get_fsdp_special_modules(self) -> typing.Dict[str, nn.Module]:
        special_modules = {}

        special_modules["token_embeddings"] = self.token_embeddings
        special_modules["lm_head"] = self.lm_head

        # MTP heads if present (they can be large with multiple prediction steps)
        if self.mtp_heads is not None:
            special_modules["mtp_heads"] = self.mtp_heads

        # Note: Not including RoPE as it has minimal parameters

        return special_modules
