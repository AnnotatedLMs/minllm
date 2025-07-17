# Standard Library
import dataclasses

# Project
from pretraining.configs.llm import head_configs
from pretraining.configs.llm import position_configs
from pretraining.configs.llm import token_configs
from pretraining.configs.llm import weight_init

# Local
from pretraining.configs.transformer import transformer_configs


@dataclasses.dataclass
class GPT2Config:
    """Configuration for GPT-2 architecture."""

    # Token embeddings (converts token IDs to vectors)
    token_embedding_config: token_configs.TokenEmbeddingConfig

    # Position embeddings (required for GPT-2)
    position_embedding_config: position_configs.LearnedPositionEmbeddingConfig

    # Transformer backbone (processes embeddings through attention + FFN layers)
    transformer_config: transformer_configs.TransformerConfig

    # Output head (projects hidden states to vocabulary)
    output_head_config: head_configs.OutputHeadConfig

    # Weight initialization
    weight_init_config: weight_init.InitializationConfig

    def __post_init__(self):
        """Validate GPT-2 configuration consistency."""
        # Ensure embedding dim matches transformer hidden dim
        if self.token_embedding_config.embedding_dim != self.transformer_config.hidden_dim:
            raise ValueError(
                f"token embedding_dim ({self.token_embedding_config.embedding_dim}) must match "
                f"transformer hidden_dim ({self.transformer_config.hidden_dim})"
            )

        # Ensure vocab sizes match
        if self.token_embedding_config.vocab_size != self.transformer_config.vocab_size:
            raise ValueError(
                f"token embedding vocab_size ({self.token_embedding_config.vocab_size}) must match "
                f"transformer vocab_size ({self.transformer_config.vocab_size})"
            )

        # Ensure position embedding dim matches transformer hidden dim
        if self.position_embedding_config.embedding_dim != self.transformer_config.hidden_dim:
            raise ValueError(
                f"position embedding_dim ({self.position_embedding_config.embedding_dim}) must match "
                f"transformer hidden_dim ({self.transformer_config.hidden_dim})"
            )

        # GPT-2 doesn't use RoPE
        if self.transformer_config.rope_config is not None:
            raise ValueError("GPT-2 architecture should not have RoPE config")


@dataclasses.dataclass
class LlamaConfig:
    """Configuration for Llama architecture (3.1 and similar)."""

    # Token embeddings (converts token IDs to vectors)
    token_embedding_config: token_configs.TokenEmbeddingConfig

    # Transformer backbone with RoPE (processes embeddings through GQA + SwiGLU layers)
    transformer_config: transformer_configs.TransformerConfig

    # Output head (projects hidden states to vocabulary)
    output_head_config: head_configs.OutputHeadConfig

    # Weight initialization
    weight_init_config: weight_init.InitializationConfig

    def __post_init__(self):
        """Validate Llama configuration consistency."""
        # Ensure embedding dim matches transformer hidden dim
        if self.token_embedding_config.embedding_dim != self.transformer_config.hidden_dim:
            raise ValueError(
                f"token embedding_dim ({self.token_embedding_config.embedding_dim}) must match "
                f"transformer hidden_dim ({self.transformer_config.hidden_dim})"
            )

        # Ensure vocab sizes match
        if self.token_embedding_config.vocab_size != self.transformer_config.vocab_size:
            raise ValueError(
                f"token embedding vocab_size ({self.token_embedding_config.vocab_size}) must match "
                f"transformer vocab_size ({self.transformer_config.vocab_size})"
            )

        # Llama requires RoPE config
        if self.transformer_config.rope_config is None:
            raise ValueError("Llama architecture requires RoPE config")


@dataclasses.dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek-V3 architecture."""

    # Token embeddings (converts token IDs to vectors)
    token_embedding_config: token_configs.TokenEmbeddingConfig

    # Transformer backbone with RoPE + MoE (processes embeddings through MLA + MoE layers)
    transformer_config: transformer_configs.TransformerConfig

    # Output head (projects hidden states to vocabulary)
    output_head_config: head_configs.OutputHeadConfig

    # Multi-token prediction heads (predicts multiple future tokens)
    mtp_config: head_configs.MultiTokenPredictionConfig

    # Weight initialization
    weight_init_config: weight_init.InitializationConfig

    def __post_init__(self):
        """Validate DeepSeek configuration consistency."""
        # Ensure embedding dim matches transformer hidden dim
        if self.token_embedding_config.embedding_dim != self.transformer_config.hidden_dim:
            raise ValueError(
                f"token embedding_dim ({self.token_embedding_config.embedding_dim}) must match "
                f"transformer hidden_dim ({self.transformer_config.hidden_dim})"
            )

        # Ensure vocab sizes match
        if self.token_embedding_config.vocab_size != self.transformer_config.vocab_size:
            raise ValueError(
                f"token embedding vocab_size ({self.token_embedding_config.vocab_size}) must match "
                f"transformer vocab_size ({self.transformer_config.vocab_size})"
            )

        # DeepSeek requires RoPE config
        if self.transformer_config.rope_config is None:
            raise ValueError("DeepSeek architecture requires RoPE config")
