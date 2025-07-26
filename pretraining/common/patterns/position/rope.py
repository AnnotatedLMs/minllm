# Third Party
import jaxtyping
import torch

# Project
from pretraining.common.patterns.position import rope_base
from pretraining.configs.model.components import position


class RoPE(rope_base.BaseRoPE):
    """
    Rotary Position Embeddings (RoPE) - the standard implementation.

    Used by: Llama 3 and similar models

    Variation: Precomputes all rotation matrices at initialization
    Computation: Stores sin/cos values for all positions up to max_seq_len
    Effect: Fast position encoding with fixed memory overhead

    The precomputation process:
    1. At init: Compute sin/cos for all positions - trades memory for speed
    2. At forward: Slice precomputed values - O(1) lookup instead of computation
    3. Apply rotation - encodes relative position through dimension pair rotation

    This approach enables efficient inference while supporting RoPE scaling
    for context length extension beyond training.
    """

    def __init__(self, dim: int, config: position.RoPEConfig, max_seq_len: int = 8192):
        """
        Initialize with precomputed frequencies.

        Precomputes rotation matrices for all positions at initialization,
        enabling O(1) position encoding during forward passes.
        """
        super().__init__(dim, config)

        # Precompute and store frequencies for efficiency
        self.max_seq_len = max_seq_len
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(max_seq_len))

    def _apply_rotation(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq heads head_dim"],
        freqs_cis: jaxtyping.Float[torch.Tensor, "seq_len dim_half 2"],
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq heads head_dim"]:
        """Apply rotation using precomputed cos/sin values - encodes position through rotation."""

        # Reshape input to separate dimension pairs
        x_shaped: jaxtyping.Float[torch.Tensor, "batch seq heads dim_half 2"]
        x_shaped = x.float().reshape(batch_size, seq_len, num_heads, -1, 2)

        # Prepare freqs_cis for broadcasting
        freqs_cis = freqs_cis.view(1, seq_len, 1, x_shaped.size(3), 2)

        # Apply rotation using complex multiplication
        x_out: jaxtyping.Float[torch.Tensor, "batch seq heads dim_half 2"]
        x_out = torch.stack(
            [
                x_shaped[..., 0] * freqs_cis[..., 0] - x_shaped[..., 1] * freqs_cis[..., 1],
                x_shaped[..., 1] * freqs_cis[..., 0] + x_shaped[..., 0] * freqs_cis[..., 1],
            ],
            dim=-1,
        )

        # Flatten back
        result: jaxtyping.Float[torch.Tensor, "batch seq heads head_dim"]
        result = x_out.flatten(3).type_as(x)

        return result

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq heads head_dim"],
        position_offset: int = 0,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq heads head_dim"]:
        """
        Apply rotary positional embeddings to full attention heads.

        Args:
            x: Input tensor [batch, seq, heads, head_dim]
               Applied to queries and keys after projection, before attention
            position_offset: Starting position for RoPE frequencies (for KV caching)

        Returns:
            Rotated tensor with same shape as input
        """
        batch_size, seq_len, num_heads, head_dim = x.shape

        # Check if we have enough precomputed frequencies
        end_pos = position_offset + seq_len
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Position range [{position_offset}, {end_pos}) exceeds precomputed max {self.max_seq_len}"
            )

        # Slice frequencies based on position offset
        freqs_cis = self.freqs_cis[position_offset:end_pos]
        return self._apply_rotation(x, freqs_cis, batch_size, seq_len, num_heads, head_dim)

    def get_freqs_cis(
        self,
        seq_len: int,
        device: torch.device,
    ) -> jaxtyping.Float[torch.Tensor, "seq_len dim_half 2"]:
        """
        Get precomputed or dynamically computed freqs_cis.

        This is useful for models that want to handle RoPE application themselves.
        """
        if seq_len <= self.max_seq_len:
            return self.freqs_cis[:seq_len].to(device)
        else:
            # Compute dynamically if beyond precomputed range
            return self._precompute_freqs_cis(seq_len).to(device)
