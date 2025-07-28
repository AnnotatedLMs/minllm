# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from pretraining.common.base import llm
from pretraining.common.patterns.blocks import gpt2 as gpt2_blocks
from pretraining.common.patterns.position import learned
from pretraining.configs.model import initialization
from pretraining.configs.model.architectures import gpt


class GPT2(llm.BaseLLM):
    """
    GPT-2 Language Model implementation.

    Used by: GPT-2

    Variation: Token embeddings + learned position embeddings
    Computation: Embeddings are added together before entering transformer
    Effect: Model learns fixed positional patterns up to max sequence length

    Variation: Uses BiasedLNTransformerBlock with standard MHA and MLP
    Computation: LayerNorm with bias, full attention heads, 4x MLP expansion
    Effect: Each head learns independent attention patterns with smooth non-linearity

    Variation: Tied embeddings for output projection
    Computation: lm_head shares weights with token_embeddings
    Effect: Reduces parameters and improves sample efficiency
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
        # Position embedding params
        max_position_embeddings: int = 1024,
        position_init_std: float = 0.02,
        # Transformer params
        num_heads: int = 12,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        use_flash_attention: bool = False,
        # Weight init params
        weight_init_std: float = 0.02,
        residual_pattern: str = "c_proj.weight",
    ):
        """
        Initialize GPT-2 model.

        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension (d_model)
            n_layers: Number of transformer layers
            block_size: Maximum sequence length
            embedding_dropout: Dropout for embeddings
            max_position_embeddings: Maximum position embeddings
            position_init_std: Std dev for position embedding init
            num_heads: Number of attention heads
            dropout: Dropout rate throughout model
            norm_eps: Layer norm epsilon
            use_flash_attention: Whether to use flash attention
            weight_init_std: Standard deviation for weight init
            residual_pattern: Pattern to identify residual connections
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

        # Position embeddings
        self.position_embeddings = learned.LearnedPositionEmbedding(
            max_position_embeddings=max_position_embeddings,
            embedding_dim=hidden_dim,
            init_std=position_init_std,
        )

        self.embed_dropout = nn.Dropout(embedding_dropout)

        # Transformer blocks
        self.blocks: nn.ModuleList[gpt2_blocks.GPT2TransformerBlock] = nn.ModuleList(
            [
                gpt2_blocks.GPT2TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    max_seq_length=block_size,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(n_layers)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(hidden_dim, eps=norm_eps, elementwise_affine=True)

        # Output projection - tied with token embeddings
        self.lm_head = self.token_embeddings

        # Apply GPT-2 weight initialization
        weight_init_config = initialization.GPT2InitConfig(
            std=weight_init_std,
            residual_pattern=residual_pattern,
            position_init_std=position_init_std,
        )
        minimal_config = type("Config", (), {"weight_init": weight_init_config})()
        self._apply_weight_initialization(minimal_config)

    @classmethod
    def from_config(cls, config: gpt.GPT2Config) -> "GPT2":
        """Create GPT2 from a config object."""
        return cls(
            # Core dimensions
            vocab_size=config.token_embedding.vocab_size,
            hidden_dim=config.transformer.hidden_dim,
            n_layers=config.transformer.n_layers,
            block_size=config.transformer.block_size,
            # Token embedding params
            embedding_dropout=config.token_embedding.embedding_dropout,
            # Position embedding params
            max_position_embeddings=config.position_embedding.max_position_embeddings,
            position_init_std=config.position_embedding.init_std,
            # Transformer params
            num_heads=config.transformer.attention.num_heads,
            dropout=config.transformer.dropout,
            norm_eps=config.transformer.normalization.norm_eps,
            use_flash_attention=config.transformer.attention.use_flash_attention,
            # Weight init params
            weight_init_std=config.weight_init.std,
            residual_pattern=config.weight_init.residual_pattern,
        )

    def _get_position_embeddings(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]:
        """Generate position embeddings for the sequence."""
        # Create position indices
        position_ids: jaxtyping.Int[torch.Tensor, "seq"]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # Expand for batch
        position_ids_expanded: jaxtyping.Int[torch.Tensor, "batch seq"]
        position_ids_expanded = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get position embeddings
        pos_embeds: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]
        pos_embeds = self.position_embeddings(position_ids_expanded)

        return pos_embeds

    def _combine_embeddings(
        self,
        token_embeds: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
        pos_embeds: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]:
        """Combine token and position embeddings."""
        combined: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]
        combined = token_embeds + pos_embeds

        # Apply dropout
        combined = self.embed_dropout(combined)

        return combined

    def _apply_transformer_blocks(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
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
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states.append(hidden)

            # Apply transformer block
            hidden = block(hidden, attention_mask=attention_mask)

        # Apply final layer norm
        normalized: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]
        normalized = self.ln_f(hidden)

        if output_hidden_states:
            all_hidden_states.append(normalized)

        return normalized, all_hidden_states

    def _compute_logits(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq vocab"]:
        """Compute output logits from hidden states."""
        if hasattr(self.lm_head, "weight"):
            # Tied embeddings case - use embedding weights transposed
            logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"]
            logits = torch.matmul(hidden_states, self.lm_head.weight.T)
        else:
            # Separate projection case
            logits = self.lm_head(hidden_states)

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
        shift_logits: jaxtyping.Float[torch.Tensor, "batch seq_minus_1 vocab"]
        shift_logits = logits[..., :-1, :].contiguous()

        shift_labels: jaxtyping.Int[torch.Tensor, "batch seq_minus_1"]
        shift_labels = targets[..., 1:].contiguous()

        # Flatten for loss computation
        flat_logits: jaxtyping.Float[torch.Tensor, "batch_x_seq vocab"]
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))

        flat_labels: jaxtyping.Int[torch.Tensor, "batch_x_seq"]
        flat_labels = shift_labels.view(-1)

        # Compute loss
        loss: torch.Tensor
        loss = nn.functional.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)

        return loss

    @torch.no_grad()
    def generate(
        self,
        idx: jaxtyping.Int[torch.Tensor, "batch seq"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: typing.Optional[int] = None,
        **kwargs,
    ) -> jaxtyping.Int[torch.Tensor, "batch new_seq"]:
        """
        Generate tokens autoregressively with GPT-2.

        Handles context window cropping when sequence exceeds block_size.

        Args:
            idx: Initial token indices [batch, seq]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k most likely tokens
            **kwargs: Additional generation arguments

        Returns:
            Generated token indices [batch, seq + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]

            # Forward the model to get logits
            logits = self.inference_forward(idx_cond, **kwargs)

            # Focus on the last token logits
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

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
            - (loss, logits, extras) if output_hidden_states=True
        """
        batch_size, seq_len = idx.shape
        device = idx.device

        # 1. Embed tokens
        token_embeds: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]
        token_embeds = self.token_embeddings(idx)

        # 2. Get position embeddings
        pos_embeds = self._get_position_embeddings(batch_size, seq_len, device)

        # 3. Combine embeddings with dropout
        x = self._combine_embeddings(token_embeds, pos_embeds)

        # 4. Apply transformer blocks
        hidden_states, all_hidden_states = self._apply_transformer_blocks(
            x, attention_mask=attention_mask, output_hidden_states=output_hidden_states
        )

        # 5. Compute logits
        logits = self._compute_logits(hidden_states)

        # 6. Compute loss
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
        attention_mask: typing.Optional[torch.Tensor] = None,
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq vocab"]:
        """
        Forward pass for inference/generation.

        Note: GPT-2 doesn't use KV cache, so this is simpler than Llama.

        Args:
            idx: Input token indices [batch, seq]
            attention_mask: Optional attention mask
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Logits tensor [batch, seq, vocab]
        """
        batch_size, seq_len = idx.shape
        device = idx.device

        # 1. Embed tokens (no dropout during inference)
        token_embeds: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]
        token_embeds = self.token_embeddings(idx)

        # 2. Get position embeddings
        pos_embeds = self._get_position_embeddings(batch_size, seq_len, device)

        # 3. Combine embeddings (no dropout)
        x = token_embeds + pos_embeds

        # 4. Apply transformer blocks
        hidden_states, _ = self._apply_transformer_blocks(
            x, attention_mask=attention_mask, output_hidden_states=False
        )

        # 5. Compute logits
        logits = self._compute_logits(hidden_states)

        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters in the model.

        Args:
            non_embedding: If True, exclude position embeddings from count

        Returns:
            Total number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.position_embeddings is not None:
            n_params -= self.position_embeddings.wpe.weight.numel()
        return n_params
