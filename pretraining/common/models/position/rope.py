# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models.position import rope_scaling
from pretraining.configs.model.components import position


class RoPEBase(nn.Module):
    """
    Base class for Rotary Position Embeddings (RoPE).

    Scholarship:
    - Su et al. RoFormer, 2021. https://arxiv.org/abs/2104.09864

    Significance:
    Encodes position through rotation rather than addition, enabling extrapolation beyond training length.
    Mathematical rotations preserve magnitude while encoding relative position through phase differences.
    Provides foundation for both precomputed and dynamic RoPE implementations.

    Init:
    Registers inverse frequencies as a buffer (non-learnable constants):
        self.register_buffer("inv_freq", inv_freq)  # Shape: [dim_half]
    These frequencies follow θ_i = 1/(theta^(2i/dim)) creating a geometric progression.
    Optional linear scaling can be configured for context extension.

    Step-by-step control flow:
    1. Initialize with dimension and base frequency theta (typically 10000)
    2. Compute inverse frequencies using geometric progression formula
    3. Store frequencies as buffer for use by subclasses
    4. Subclasses use these to generate sin/cos values for rotation
    5. Apply 2D rotations to dimension pairs during forward pass

    Learning process:
    - This module contains no learnable parameters.
    - The inverse frequencies are fixed mathematical constants derived from theta and dimensions.
    - Rotations are deterministic transformations based on position indices.

    - Architectural benefit:
        - Rotation preserves vector magnitude unlike addition, maintaining semantic strength
        - Position encoded through phase rather than magnitude prevents gradient instability
        - Orthogonal transformation ensures stable training dynamics
        - Enables natural extrapolation beyond training length (rotation patterns are consistent)
        - Relative position emerges from phase difference in dot product (no explicit modeling needed)
        - Position and content remain separable throughout the network
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        linear_scaling: typing.Optional[position.LinearRoPEScalingConfig] = None,
    ):
        """
        Initialize base RoPE.

        Args:
            dim: Dimension to apply RoPE to (must be even)
            theta: Base frequency for RoPE
            scaling: Optional RoPE scaling configuration
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.linear_scaling = linear_scaling

        # Compute and store inverse frequencies
        inv_freq: jaxtyping.Float[torch.Tensor, "dim_half"]
        inv_freq = self._compute_inv_freq(dim, theta)
        # Buffer: precomputed frequencies for RoPE - fixed throughout training
        # Not a parameter since these are mathematical constants, not learned
        self.register_buffer("inv_freq", inv_freq)

    def _compute_inv_freq(
        self,
        dim: int,
        theta: float,
    ) -> jaxtyping.Float[torch.Tensor, "dim_half"]:
        """
        Compute inverse frequencies for RoPE.

        The formula: freq_i = 1 / (theta^(2i/dim)) for i in [0, dim//2)

        This creates a geometric progression of frequencies, similar to
        Fourier features, allowing the model to attend to different
        "wavelengths" of positional patterns.
        """
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        return inv_freq

    def _precompute_freqs_cis(
        self,
        seq_len: int,
        inv_freq: typing.Optional[torch.Tensor] = None,
    ) -> jaxtyping.Float[torch.Tensor, "seq_len dim_half 2"]:
        """Precompute cos and sin values for all positions."""
        # Use provided frequencies or default
        if inv_freq is None:
            inv_freq = self.inv_freq

        # Apply linear scaling if configured (Llama-style)
        if self.linear_scaling is not None:
            inv_freq = rope_scaling.apply_rope_scaling(inv_freq, self.linear_scaling)

        # Create position indices
        positions: jaxtyping.Float[torch.Tensor, "seq_len"]
        positions = torch.arange(seq_len, dtype=torch.float32, device=inv_freq.device)

        # Compute outer product: position × frequency
        freqs_outer: jaxtyping.Float[torch.Tensor, "seq_len dim_half"]
        freqs_outer = torch.outer(positions, inv_freq)

        # Convert to complex exponentials then extract real/imag
        freqs_complex = torch.polar(torch.ones_like(freqs_outer), freqs_outer)

        freqs_cis: jaxtyping.Float[torch.Tensor, "seq_len dim_half 2"]
        freqs_cis = torch.stack([freqs_complex.real, freqs_complex.imag], dim=-1)

        return freqs_cis

    def _compute_dynamic_freqs(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        inv_freq: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "seq_len dim_half"],
        jaxtyping.Float[torch.Tensor, "seq_len dim_half"],
    ]:
        """Compute sin/cos dynamically for the given sequence length."""
        # Use provided frequencies or default
        if inv_freq is None:
            inv_freq = self.inv_freq.to(device)
        else:
            inv_freq = inv_freq.to(device)

        # Apply linear scaling if configured (Llama-style)
        if self.linear_scaling is not None:
            inv_freq = rope_scaling.apply_rope_scaling(inv_freq, self.linear_scaling)

        # Create positions
        positions: jaxtyping.Float[torch.Tensor, "seq_len"]
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)

        # Compute position × frequency
        sinusoid: jaxtyping.Float[torch.Tensor, "seq_len dim_half"]
        sinusoid = positions[:, None] * inv_freq[None, :]

        # Get sin and cos
        sin: jaxtyping.Float[torch.Tensor, "seq_len dim_half"] = torch.sin(sinusoid).to(dtype)
        cos: jaxtyping.Float[torch.Tensor, "seq_len dim_half"] = torch.cos(sinusoid).to(dtype)

        return cos, sin


class PrecomputedRoPE(RoPEBase):
    """
    Rotary Position Embeddings (RoPE) - the standard precomputed implementation.

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

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        max_seq_len: int = 8192,
        linear_scaling: typing.Optional[position.LinearRoPEScalingConfig] = None,
    ):
        """
        Initialize with precomputed frequencies.

        Precomputes rotation matrices for all positions at initialization,
        enabling O(1) position encoding during forward passes.

        Args:
            dim: Dimension to apply RoPE to (must be even)
            theta: Base frequency for RoPE
            max_seq_len: Maximum sequence length to precompute
            scaling: Optional RoPE scaling configuration for extended context
        """
        super().__init__(dim, theta=theta, linear_scaling=linear_scaling)

        # Precompute and store frequencies for efficiency
        self.max_seq_len = max_seq_len
        # Buffer: precomputed complex exponentials to avoid recomputation
        # Stored as buffer for efficiency - computing these on every forward is expensive
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
        # Store original dtype
        orig_dtype = x.dtype

        # Reshape input to separate dimension pairs (compute in float32 for accuracy)
        x_shaped: jaxtyping.Float[torch.Tensor, "batch seq heads dim_half 2"]
        x_shaped = x.float().reshape(batch_size, seq_len, num_heads, -1, 2)

        # Prepare freqs_cis for broadcasting (ensure float32)
        freqs_cis = freqs_cis.float().view(1, seq_len, 1, x_shaped.size(3), 2)

        # Apply rotation using complex multiplication
        x_out: jaxtyping.Float[torch.Tensor, "batch seq heads dim_half 2"]
        x_out = torch.stack(
            [
                x_shaped[..., 0] * freqs_cis[..., 0] - x_shaped[..., 1] * freqs_cis[..., 1],
                x_shaped[..., 1] * freqs_cis[..., 0] + x_shaped[..., 0] * freqs_cis[..., 1],
            ],
            dim=-1,
        )

        # Flatten back and restore original dtype
        rotated_flat: jaxtyping.Float[torch.Tensor, "batch seq heads head_dim"]
        rotated_flat = x_out.flatten(3).to(orig_dtype)

        return rotated_flat

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
