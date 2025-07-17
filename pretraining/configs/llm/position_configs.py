# Standard Library
import dataclasses


@dataclasses.dataclass
class LearnedPositionEmbeddingConfig:
    """Configuration for learned position embeddings (GPT-2 style)."""

    max_position_embeddings: int = 1024
    embedding_dim: int = 768  # Must match model hidden dim

    # Position embedding initialization
    init_std: float = 0.02  # Standard deviation for normal initialization

    def __post_init__(self):
        """Validate position embedding configuration."""
        if self.max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {self.max_position_embeddings}"
            )
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
