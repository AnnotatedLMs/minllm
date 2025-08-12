# Third Party
import pydantic

# Project
from pretraining.configs import base


class TokenEmbeddingConfig(base.BaseConfig):
    """Configuration for token embedding layers."""

    embedding_dim: int = pydantic.Field(gt=0, description="Embedding dimension")

    # Embedding initialization
    init_std: float = pydantic.Field(
        gt=0, description="Standard deviation for normal initialization"
    )


class LearnedPositionEmbeddingConfig(base.BaseConfig):
    """Configuration for learned position embeddings (GPT-2 style)."""

    max_position_embeddings: int = pydantic.Field(default=1024, gt=0)
    embedding_dim: int = pydantic.Field(default=768, gt=0)  # Must match model hidden dim

    # Position embedding initialization
    init_std: float = pydantic.Field(
        default=0.02, gt=0, description="Standard deviation for normal initialization"
    )
