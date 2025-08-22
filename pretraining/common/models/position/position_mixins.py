# Standard Library

# Third Party
import jaxtyping
import torch

# Project
from pretraining.common.models.position import partial_rope


class PartialRoPEApplicationMixin:
    """
    Mixin for applying RoPE to a subset of dimensions (decoupled RoPE strategy).

    Scholarship:
    Deepseek-V2, 2024, https://arxiv.org/pdf/2405.04434
    Deepseek-V3, 2025, https://arxiv.org/pdf/2412.19437

    Significance:
    Solves the incompatibility between RoPE and low-rank KV compression.
    Allows position information to bypass compression, preserving relative position accuracy.

    Init:
    This mixin has no initialization. It works with components defined in MultiHeadLatentAttention:
        self.key_rope_projection = nn.Linear(hidden_dim, num_heads * rope_dim)  # Creates position features from input
        self.query_rope_projection = nn.Linear(query_compression_dim, num_heads * rope_dim)  # Creates position features from compressed
        self.key_rope_rotation = PartialRoPE(rope_dim)  # Applies rotation to key position features
        self.query_rope_rotation = PartialRoPE(rope_dim)  # Applies rotation to query position features

    Step-by-step control flow (_apply_partial_rope):
    1. Receive already-separated content and position features
    2. Transpose position features for RoPE module format
    3. Apply rotary embeddings to position features
    4. Concatenate untouched content with rotated position features
    5. Return combined features for attention computation

    Learning process:
    - This mixin contains no learnable parameters.
    - The PartialRoPE module it uses is also non-parameterized - it applies fixed rotations based on mathematical formulas.

    - Decoupling benefit:
      - Position and content optimize independently through separate gradient paths
      - RoPE rotation only affects position dimensions, leaving content semantics intact
      - Allows aggressive KV compression while maintaining precise position awareness
      - During inference, position features can be cached separately from compressed content
    """

    def _apply_partial_rope(
        self,
        content_features: jaxtyping.Float[torch.Tensor, "batch seq_len heads content_dim"],
        position_features: jaxtyping.Float[torch.Tensor, "batch seq_len heads rope_dim"],
        rope_module: partial_rope.DecoupledRoPE,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len heads full_dim"]:
        """Apply RoPE to position features and concatenate with content."""
        batch_size, seq_len, num_heads, rope_dim = position_features.shape

        # Reshape for RoPE: [batch, heads, seq, rope_dim]
        position_reshaped: jaxtyping.Float[torch.Tensor, "batch heads seq rope_dim"]
        position_reshaped = position_features.transpose(1, 2)

        # Apply RoPE rotation
        position_rotated: jaxtyping.Float[torch.Tensor, "batch heads seq rope_dim"]
        position_rotated = rope_module(position_reshaped)

        # Reshape back: [batch, seq, heads, rope_dim]
        position_rotated = position_rotated.transpose(1, 2)

        # Concatenate content and rotated position
        combined: jaxtyping.Float[torch.Tensor, "batch seq_len heads full_dim"]
        combined = torch.cat([content_features, position_rotated], dim=-1)

        return combined
