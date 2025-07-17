# Standard Library
import dataclasses
import typing


@dataclasses.dataclass
class BaseAttentionConfig:
    """Base configuration for all attention mechanisms."""

    num_heads: int
    dropout: float
    bias: bool
    max_seq_length: int  # For GPT-2 style pre-computed bias
    is_causal: bool
    attention_type: typing.Literal["multi_head", "grouped_query", "multi_head_latent"]
    use_flash_attention: bool  # Whether to use Flash Attention when available


@dataclasses.dataclass
class MultiHeadAttentionConfig(BaseAttentionConfig):
    """Configuration for standard multi-head attention (GPT-2)."""

    def __post_init__(self):
        self.attention_type = "multi_head"


@dataclasses.dataclass
class GroupedQueryAttentionConfig(BaseAttentionConfig):
    """Configuration for grouped query attention (Llama 3 70B)."""

    num_kv_heads: int

    def __post_init__(self):
        self.attention_type = "grouped_query"
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) cannot exceed num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )


@dataclasses.dataclass
class MultiHeadLatentAttentionConfig(BaseAttentionConfig):
    """Configuration for multi-head latent attention (DeepSeek-V3)."""

    head_dim: int  # Explicit head dimension for MLA
    kv_compression_dim: int
    query_compression_dim: int
    rope_dim: int  # MLA uses separate RoPE dimension

    def __post_init__(self):
        self.attention_type = "multi_head_latent"
