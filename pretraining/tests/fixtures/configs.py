"""
Shared test configurations for unit and integration tests.
"""

# Project
# Local
from pretraining.configs.transformer import attention_configs
from pretraining.configs.transformer import ffn_configs
from pretraining.configs.transformer import normalization_configs
from pretraining.configs.transformer import position_configs
from pretraining.configs.transformer import transformer_configs


def create_test_attention_config(
    num_heads: int = 4,
    hidden_dim: int = 128,
) -> attention_configs.AttentionConfig:
    """Create a small attention config for testing."""
    return attention_configs.AttentionConfig(
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
    )


def create_test_gqa_config(
    num_heads: int = 8,
    num_kv_heads: int = 2,
) -> attention_configs.GroupedQueryAttentionConfig:
    """Create a GQA config for testing."""
    return attention_configs.GroupedQueryAttentionConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
        bias=False,
    )


def create_test_rope_config() -> position_configs.RoPEConfig:
    """Create a RoPE config for testing."""
    return position_configs.RoPEConfig(
        theta=10000.0,
        scaling=None,
    )


def create_test_transformer_config(
    hidden_dim: int = 128,
    n_layers: int = 2,
    block_size: int = 512,
) -> transformer_configs.TransformerConfig:
    """Create a small transformer config for testing."""
    return transformer_configs.TransformerConfig(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        block_size=block_size,
        vocab_size=1000,  # Small vocab for testing
        dropout=0.0,
        bias=True,
        attention_config=create_test_attention_config(hidden_dim=hidden_dim),
        ffn_config=ffn_configs.FFNConfig(
            activation="gelu",
            dropout=0.0,
            bias=True,
        ),
        normalization_config=normalization_configs.NormalizationConfig(
            norm_type="layernorm",
            norm_eps=1e-5,
        ),
        rope_config=None,  # GPT-2 style doesn't use RoPE
    )
