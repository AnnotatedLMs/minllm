# Standard Library
import typing

# Third Party
import jaxtyping
import torch


class PartialRoPEApplicationMixin:
    """
    Mixin for applying RoPE to a subset of dimensions.

    Variation: Applies RoPE only to position-specific features
    Computation: Rotates position dimensions, concatenates with content
    Effect: Separates position encoding from semantic content

    Used by: DeepSeek-V3's Multi-head Latent Attention
    """

    def _apply_partial_rope(
        self,
        content_features: jaxtyping.Float[torch.Tensor, "batch seq_len heads content_dim"],
        position_features: jaxtyping.Float[torch.Tensor, "batch seq_len heads rope_dim"],
        rope_module: typing.Any,  # Will be PartialRoPE instance
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

    def _split_content_and_position(
        self,
        features: jaxtyping.Float[torch.Tensor, "batch seq_len heads full_dim"],
        rope_dim: int,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq_len heads content_dim"],
        jaxtyping.Float[torch.Tensor, "batch seq_len heads rope_dim"],
    ]:
        """Split features into content and position components."""
        content_dim = features.shape[-1] - rope_dim

        content: jaxtyping.Float[torch.Tensor, "batch seq_len heads content_dim"]
        position: jaxtyping.Float[torch.Tensor, "batch seq_len heads rope_dim"]
        content, position = features.split([content_dim, rope_dim], dim=-1)

        return content, position
