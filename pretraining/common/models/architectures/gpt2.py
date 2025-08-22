# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.mixins import fsdp_mixins
from pretraining.common.mixins import generation_mixins
from pretraining.common.mixins import weight_init_mixins
from pretraining.common.models import outputs
from pretraining.common.models.blocks import gpt2_blocks
from pretraining.common.models.embeddings import embedding_mixins
from pretraining.common.models.position import learned
from pretraining.configs.model.architectures import gpt


class GPT2(
    nn.Module,
    embedding_mixins.PositionIDGenerationMixin,
    fsdp_mixins.FSDPMixin,
    weight_init_mixins.TransformerWeightInitMixin,
    generation_mixins.AutoregressiveGenerationMixin,
):
    """
    GPT-2 language model - Autoregressive transformer with learned position embeddings.
    Radford et al., 2019, https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

    Step-by-step control flow (model-level orchestration):
    1. Token embedding: Convert token IDs to dense vectors
    2. Position embedding: Add learned position embeddings
    3. Embedding dropout: Apply dropout for regularization
    4. Transformer blocks: N layers of GPT2TransformerBlock
    5. Final LayerNorm: Normalize final hidden states
    6. Output projection: Use tied embeddings (transposed) for logits

    Learning components (what learns across the model):
    - Token embeddings: Learn semantic representations for each token
    - Position embeddings: Learn absolute position patterns up to max_seq_len
    - GPT2TransformerBlocks: Each contains learnable MHA and MLP layers
    - Final LayerNorm: Learns scale and bias for output normalization
    - Weight tying: Token embeddings serve dual purpose (input and output)

    Key architectural choices:
    - Learned position embeddings: Fixed max length but learnable patterns
    - Weight tying: Reduces parameters by ~30M for 50K vocab
    - Biases everywhere: Additional flexibility vs modern bias-free designs
    - Pre-norm architecture: Better gradient flow than post-norm
    - GELU activation: Smoother than ReLU for language modeling

    Mixin contributions:
    - PositionIDGenerationMixin: Creates position IDs for learned embeddings
    - TransformerWeightInitMixin: Scaled init with special residual handling
    - AutoregressiveGenerationMixin: Standard left-to-right generation
    - FSDPMixin: Enables efficient distributed training
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
        # FFN params
        ffn_activation: str = "gelu",
        # Weight init params
        weight_init_std: float = 0.02,
        residual_pattern: str = "down_proj.weight",
    ):
        """Initialize GPT-2 model."""
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.block_size = block_size

        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
        )

        self.position_embeddings = learned.LearnedPositionEmbedding(
            max_position_embeddings=max_position_embeddings,
            embedding_dim=hidden_dim,
            init_std=position_init_std,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks: nn.ModuleList[gpt2_blocks.GPT2TransformerBlock] = nn.ModuleList(
            [
                gpt2_blocks.GPT2TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    max_seq_length=block_size,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                    ffn_activation=ffn_activation,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(hidden_dim, eps=norm_eps, elementwise_affine=True)

        # Weight Tying
        self.lm_head = self.token_embeddings

        self._apply_weight_initialization(
            n_layers=n_layers,
            std=weight_init_std,
            residual_pattern=residual_pattern,
            position_embeddings=self.position_embeddings,
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
            # FFN params
            ffn_activation=config.transformer.ffn.activation,
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
        position_ids = self._get_position_ids(batch_size, seq_len, device)
        return self.position_embeddings(position_ids)

    def forward(
        self,
        input_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> outputs.ForwardOutput:
        """
        Process input tokens through the GPT-2 model.

        The GPT-2 forward pass:
        1. Token embeddings - convert token IDs to dense vectors
        2. Position embeddings - add learned positional information
        3. Combine embeddings - sum token and position embeddings, apply dropout
        4. Transformer blocks - apply N layers of self-attention and FFN
        5. Final layer norm - normalize the final hidden states
        6. Output projection - project to vocabulary size using tied embeddings
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        token_embeds = self.token_embeddings(input_ids)
        pos_embeds = self._get_position_embeddings(batch_size, seq_len, device)
        x = token_embeds + pos_embeds
        x = self.embedding_dropout(x)

        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"] = x
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.ln_f(hidden_states)

        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"]
        logits = torch.matmul(hidden_states, self.lm_head.weight.T)

        return outputs.ForwardOutput(logits=logits)

    def generate(
        self,
        token_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: typing.Optional[int] = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> jaxtyping.Int[torch.Tensor, "batch new_seq"]:
        """Generate tokens using the mixin."""
        return super().generate(
            model_forward=self.forward,
            block_size=self.block_size,
            token_ids=token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            attention_mask=attention_mask,
        )

    def get_fsdp_wrappable_modules(self) -> typing.Set[typing.Type[nn.Module]]:
        """Return module types that FSDP should wrap."""
        return {gpt2_blocks.GPT2TransformerBlock}

    def get_transformer_blocks(self) -> typing.List[nn.Module]:
        """Return transformer block instances."""
        return list(self.blocks)

    def get_fsdp_special_modules(self) -> typing.Dict[str, nn.Module]:
        """Return special modules for FSDP wrapping."""
        return super().get_fsdp_special_modules(
            token_embeddings=self.token_embeddings,
            position_embeddings=self.position_embeddings,
            lm_head=self.lm_head,
        )
