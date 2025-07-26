"""
Shared test configurations for unit and integration tests.
"""

# Project

# Project
from pretraining.configs.model import transformer
from pretraining.configs.model.components import attention
from pretraining.configs.model.components import feedforward
from pretraining.configs.model.components import normalization
from pretraining.configs.model.components import position


def create_test_attention_config(
    num_heads: int = 4,
    hidden_dim: int = 128,
) -> attention.AttentionConfig:
    """Create a small attention config for testing."""
    return attention.AttentionConfig(
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
    )


def create_test_gqa_config(
    num_heads: int = 8,
    num_kv_heads: int = 2,
) -> attention.GroupedQueryAttentionConfig:
    """Create a GQA config for testing."""
    return attention.GroupedQueryAttentionConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
        bias=False,
    )


def create_test_rope_config() -> position.RoPEConfig:
    """Create a RoPE config for testing."""
    return position.RoPEConfig(
        theta=10000.0,
        scaling=None,
    )


def create_test_transformer_config(
    hidden_dim: int = 128,
    n_layers: int = 2,
    block_size: int = 512,
) -> transformer.TransformerConfig:
    """Create a small transformer config for testing."""
    return transformer.TransformerConfig(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        block_size=block_size,
        vocab_size=1000,  # Small vocab for testing
        dropout=0.0,
        bias=True,
        attention=create_test_attention_config(hidden_dim=hidden_dim),
        ffn=feedforward.FFNConfig(
            activation="gelu",
            dropout=0.0,
            bias=True,
        ),
        normalization=normalization.NormalizationConfig(
            norm_type="layernorm",
            norm_eps=1e-5,
        ),
        rope=None,  # GPT-2 style doesn't use RoPE
    )
