# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.base import llm
from pretraining.common.base import outputs
from pretraining.common.patterns.blocks import gpt2 as gpt2_blocks
from pretraining.common.patterns.position import learned
from pretraining.configs.model.architectures import gpt
from pretraining.utils import weight_init
from pretraining.utils.training import fsdp_policies


class GPT2(llm.BaseLLM, fsdp_policies.FSDPMixin):
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
        # Position embedding params
        max_position_embeddings: int = 1024,
        position_init_std: float = 0.02,
        # Transformer params
        num_heads: int = 12,
        dropout: typing.Optional[float] = None,
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
            hidden_dim: Hidden dimension (hidden_dim)
            n_layers: Number of transformer layers
            block_size: Maximum sequence length
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

        self.embedding_dropout = nn.Dropout(dropout) if dropout is not None else None

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

        # Weight tying: Output projection shares weights with input embeddings
        # This means the same matrix is used for both token→embedding and hidden→logits
        # Saves parameters (50M+ for large vocabs) and improves training efficiency
        self.lm_head = self.token_embeddings

        self._apply_weight_initialization(
            std=weight_init_std,
            residual_pattern=residual_pattern,
            position_init_std=position_init_std,
        )

    @classmethod
    def from_config(cls, config: gpt.GPT2Config) -> "GPT2":
        """Create GPT2 from a config object."""
        return cls(
            # Core dimensions
            vocab_size=config.vocab_size,
            hidden_dim=config.transformer.hidden_dim,
            n_layers=config.transformer.n_layers,
            block_size=config.transformer.block_size,
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
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Generate position embeddings for the sequence."""
        # Create position indices
        position_ids: jaxtyping.Int[torch.Tensor, "seq"]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # Expand for batch
        position_ids_expanded: jaxtyping.Int[torch.Tensor, "batch seq"]
        position_ids_expanded = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get position embeddings
        pos_embeds: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        pos_embeds = self.position_embeddings(position_ids_expanded)

        return pos_embeds

    def _combine_embeddings(
        self,
        token_embeds: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        pos_embeds: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Combine token and position embeddings."""
        combined: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        combined = token_embeds + pos_embeds
        return combined

    def _apply_transformer_blocks(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        attention_mask: typing.Optional[torch.Tensor] = None,
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
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states.append(hidden)

            # Apply transformer block
            hidden = block(hidden, attention_mask=attention_mask)

        # Apply final layer norm
        normalized: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        normalized = self.ln_f(hidden)

        if output_hidden_states:
            all_hidden_states.append(normalized)

        return normalized, all_hidden_states

    def _compute_logits(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len vocab"]:
        """Compute output logits from hidden states."""
        # Tied embeddings: hidden states are projected to vocabulary space using
        # the same weight matrix as token embeddings (transposed)
        logits: jaxtyping.Float[torch.Tensor, "batch seq_len vocab"]
        logits = torch.matmul(hidden_states, self.lm_head.weight.T)
        return logits

    def _apply_weight_initialization(
        self,
        std: float,
        residual_pattern: str,
        position_init_std: float,
    ) -> None:
        """Apply GPT-2 specific weight initialization.

        Args:
            std: Standard deviation for general weight init
            residual_pattern: Pattern to identify residual connections
            position_init_std: Standard deviation for position embedding init
        """

        # Create and apply initializer
        initializer = weight_init.TransformerWeightInitializer(
            n_layer=self.n_layers,
            std=std,
            residual_pattern=residual_pattern,
        )
        initializer.initialize(self)

        # Position embeddings use different initialization standard from other weights
        nn.init.normal_(self.position_embeddings.wpe.weight, std=position_init_std)

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
        Generate tokens autoregressively with GPT-2.

        Handles context window cropping when sequence exceeds block_size.

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

    def forward(
        self,
        input_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> outputs.ForwardOutput:
        """
        Process input tokens through the GPT-2 model.

        The GPT-2 forward pass:
        1. Token embeddings - convert token IDs to dense vectors
        2. Position embeddings - add learned positional information (fixed up to max_position_embeddings)
        3. Combine embeddings - sum token and position embeddings, apply dropout
        4. Transformer blocks - apply N layers of self-attention and feed-forward networks
        5. Final layer norm - normalize the final hidden states
        6. Output projection - project to vocabulary size using tied embeddings

        Note: GPT-2 uses learned position embeddings (not RoPE), limiting sequence length to training max.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        token_embeds: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        token_embeds = self.token_embeddings(input_ids)

        pos_embeds = self._get_position_embeddings(batch_size, seq_len, device)

        x = self._combine_embeddings(token_embeds, pos_embeds)
        x = self._maybe_apply_dropout(x, self.embedding_dropout)

        hidden_states, _ = self._apply_transformer_blocks(
            x, attention_mask=attention_mask, output_hidden_states=False
        )

        logits = self._compute_logits(hidden_states)

        return outputs.ForwardOutput(logits=logits)

    def get_fsdp_wrappable_modules(self) -> typing.Set[typing.Type[nn.Module]]:
        return {gpt2_blocks.GPT2TransformerBlock}

    def get_transformer_blocks(self) -> typing.List[nn.Module]:
        return list(self.blocks)

    def get_fsdp_special_modules(self) -> typing.Dict[str, nn.Module]:
        special_modules = {}
        special_modules["token_embeddings"] = self.token_embeddings
        special_modules["position_embeddings"] = self.position_embeddings

        # NOT including lm_head because: self.lm_head = self.token_embeddings (weight tied)

        return special_modules
