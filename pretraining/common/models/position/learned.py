# Third Party
import jaxtyping
import torch
from torch import nn


class LearnedPositionEmbedding(nn.Module):
    """
    Learned positional embeddings (GPT-2 style).

    Used by: GPT-2 and early transformer models

    Variation: Each position has its own learned embedding vector
    Computation: Simple lookup table with learnable parameters
    Effect: Model learns fixed positional patterns limited to training length

    This approach works well for fixed-length sequences but cannot extrapolate
    to positions beyond max_position_embeddings seen during training.
    """

    def __init__(
        self,
        max_position_embeddings: int,
        embedding_dim: int,
        init_std: float = 0.02,
    ):
        """
        Initialize learned position embeddings.

        Creates a lookup table where each position index maps to a learned vector,
        allowing the model to develop position-specific representations.
        """
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.embedding_dim = embedding_dim

        # GPT-2 naming: weight-position-embedding
        self.wpe = nn.Embedding(max_position_embeddings, embedding_dim)

        # Initialize weights
        nn.init.normal_(self.wpe.weight, mean=0.0, std=init_std)

    def forward(
        self,
        position_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq embedding_dim"]:
        """
        Get position embeddings for given positions.

        Simple lookup operation that retrieves learned vectors for each position,
        enabling the model to distinguish token positions in the sequence.
        """
        if (position_ids >= self.max_position_embeddings).any():
            raise ValueError(
                f"Position ids must be less than max_position_embeddings "
                f"({self.max_position_embeddings})"
            )

        position_embeds: jaxtyping.Float[torch.Tensor, "batch seq embedding_dim"]
        position_embeds = self.wpe(position_ids)

        return position_embeds
