# Standard Library
import dataclasses
import typing

# Project
from pretraining.configs.transformer import attention_configs
from pretraining.configs.transformer import ffn_configs
from pretraining.configs.transformer import moe_configs
from pretraining.configs.transformer import normalization_configs
from pretraining.configs.transformer import position_configs


@dataclasses.dataclass
class TransformerConfig:
    """
    Configuration for transformers used in llms.
    """

    # 1. Core dimensions (define the model size and capacity)
    vocab_size: int  # Size of the token vocabulary
    hidden_dim: int  # Dimension of hidden states (d_model)
    n_layers: int  # Number of transformer blocks
    block_size: int  # Maximum sequence length

    # 2. Normalization (applied before/after each sub-layer)
    normalization_config: normalization_configs.NormalizationConfig

    # 3. Attention mechanism (core sequence processing)
    attention_config: typing.Union[
        attention_configs.MultiHeadAttentionConfig,  # Standard MHA (GPT-2)
        attention_configs.GroupedQueryAttentionConfig,  # GQA (Llama)
        attention_configs.MultiHeadLatentAttentionConfig,  # MLA (DeepSeek)
    ]

    # 4. Position encoding (optional - used within attention for position-aware processing)
    rope_config: typing.Optional[position_configs.RoPEConfig] = None

    # 5. Feed-forward network (choose one: FFN or MoE)
    ffn_config: typing.Optional[ffn_configs.FFNConfig] = None  # Standard FFN
    moe_config: typing.Optional[moe_configs.MoEConfig] = None  # Mixture of Experts

    # 6. Global parameters (affect entire model)
    dropout: float = 0.0  # Dropout rate applied throughout
    bias: bool = True  # Whether to use bias in linear layers

    def __post_init__(self):
        """Validate transformer configuration."""
        # Validate core dimensions
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")

        # Validate dropout
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        # Validate FFN/MoE configuration
        if self.ffn_config is not None and self.moe_config is not None:
            raise ValueError("Cannot have both ffn_config and moe_config")

        if self.ffn_config is None and self.moe_config is None:
            raise ValueError("Must have either ffn_config or moe_config")
