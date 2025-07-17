# Standard Library
import dataclasses


@dataclasses.dataclass
class TokenEmbeddingConfig:
    """Configuration for token embedding layers."""

    vocab_size: int
    embedding_dim: int
    embedding_dropout: float

    # Embedding initialization
    init_std: float  # Standard deviation for normal initialization

    def __post_init__(self):
        """Validate token embedding configuration."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if not 0 <= self.embedding_dropout < 1:
            raise ValueError(f"embedding_dropout must be in [0, 1), got {self.embedding_dropout}")
