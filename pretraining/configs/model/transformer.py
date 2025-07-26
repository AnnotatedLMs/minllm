# Standard Library
import typing

# Third Party
import pydantic

# Project
from pretraining.configs import base
from pretraining.configs.model.components import attention
from pretraining.configs.model.components import feedforward
from pretraining.configs.model.components import normalization
from pretraining.configs.model.components import position


class TransformerConfig(base.BaseConfig):
    """
    Configuration for transformers used in llms.
    """

    # 1. Core dimensions (define the model size and capacity)
    vocab_size: int = pydantic.Field(gt=0, description="Size of the token vocabulary")
    hidden_dim: int = pydantic.Field(gt=0, description="Dimension of hidden states (d_model)")
    n_layers: int = pydantic.Field(gt=0, description="Number of transformer blocks")
    block_size: int = pydantic.Field(gt=0, description="Maximum sequence length")

    # 2. Normalization (applied before/after each sub-layer)
    normalization: normalization.BaseNormalizationConfig

    # 3. Attention mechanism (core sequence processing)
    attention: attention.BaseAttentionConfig

    # 4. Position encoding (optional - used within attention for position-aware processing)
    rope: typing.Optional[position.RoPEConfig] = None

    # 5. Feed-forward network (choose one: FFN or MoE)
    ffn: typing.Optional[feedforward.FFNConfig] = None  # Standard FFN
    moe: typing.Optional[feedforward.MoEConfig] = None  # Mixture of Experts

    # 6. Global parameters (affect entire model)
    dropout: float = pydantic.Field(
        default=0.0, ge=0, lt=1, description="Dropout rate applied throughout"
    )
    bias: bool = True  # Whether to use bias in linear layers

    @pydantic.model_validator(mode="after")
    def validate_ffn_or_moe(self):
        """Validate FFN/MoE configuration."""
        if self.ffn is not None and self.moe is not None:
            raise ValueError("Cannot have both ffn and moe")

        if self.ffn is None and self.moe is None:
            raise ValueError("Must have either ffn or moe")

        return self
