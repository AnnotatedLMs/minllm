# Third Party
import jaxtyping
import torch

# Project
from pretraining.common.patterns.position import rope_base


class PartialRoPE(rope_base.BaseRoPE):
    """
    RoPE applied to only a subset of head dimensions.

    Used by: DeepSeek-V3's Multi-head Latent Attention (MLA)

    Variation: Applies RoPE to only rope_dim dimensions (e.g., 64 out of 128)
    Computation: Dynamic frequency calculation for position-only features
    Effect: Model learns to separate semantic content from positional information

    The process:
    1. MLA creates k_rope and q_rope via separate linear projections - these vectors
       ONLY encode position, they start with no semantic information
    2. Generate rotation frequencies - position 0 stays unchanged, position 1 rotates
       by θ, position 2 by 2θ, etc. Different dimensions rotate at different speeds
    3. Apply rotations - multiply vectors by cos/sin values, literally rotating them
       in 2D planes. This encodes position as rotation angle
    4. Later, MLA concatenates rotated position vectors with unrotated content vectors

    Why rotation encodes position: After rotation, the dot product between positions
    i and j depends only on (i-j), not absolute positions. This relative encoding
    is what transformers need for attention patterns.
    """

    def _apply_rotation(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch heads seq rope_dim"],
        cos: jaxtyping.Float[torch.Tensor, "batch heads seq dim_half"],
        sin: jaxtyping.Float[torch.Tensor, "batch heads seq dim_half"],
        batch_size: int,
        num_heads: int,
        seq_len: int,
        rope_dim: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch heads seq rope_dim"]:
        """Apply rotation in DeepSeek format."""

        # Reshape to separate dimension pairs
        x_rot: jaxtyping.Float[torch.Tensor, "batch heads seq dim_half 2"]
        x_rot = x.view(batch_size, num_heads, seq_len, rope_dim // 2, 2)

        x1: jaxtyping.Float[torch.Tensor, "batch heads seq dim_half"]
        x2: jaxtyping.Float[torch.Tensor, "batch heads seq dim_half"]
        x1, x2 = x_rot.unbind(dim=-1)

        # Apply rotation
        rotated: jaxtyping.Float[torch.Tensor, "batch heads seq dim_half 2"]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        # Reshape back
        result: jaxtyping.Float[torch.Tensor, "batch heads seq rope_dim"]
        result = rotated.view(batch_size, num_heads, seq_len, rope_dim)

        return result

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch heads seq rope_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch heads seq rope_dim"]:
        """
        Apply rotary positional embeddings to position-only vectors.

        Args:
            x: Input tensor [batch, heads, seq, rope_dim]
               Applied to k_R and q_R (position-only projections)

        Returns:
            Rotated tensor with same shape as input
        """
        batch_size, num_heads, seq_len, rope_dim = x.shape

        # Compute sin/cos dynamically
        cos, sin = self._compute_dynamic_freqs(seq_len, x.device, x.dtype)

        # Expand cos/sin for all heads and batch
        cos = cos[None, None, :, :].expand(batch_size, num_heads, -1, -1)
        sin = sin[None, None, :, :].expand(batch_size, num_heads, -1, -1)

        # Apply rotation
        return self._apply_rotation(x, cos, sin, batch_size, num_heads, seq_len, rope_dim)
