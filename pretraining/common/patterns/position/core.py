# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from pretraining.common.patterns.position import rope_scaling
from pretraining.configs.model.components import position


class RoPEBase(nn.Module):
    """
    Base class for Rotary Position Embeddings (RoPE).

    Core RoPE concepts:
    - Position encoded as rotation angle: pos 0 = 0°, pos 1 = θ, pos 2 = 2θ
    - Different dimension pairs rotate at different frequencies
    - After rotation, dot product depends only on relative position (i-j)
    - Unlike learned embeddings, RoPE can extrapolate beyond training length

    The rotation mathematics:
    1. Split dimensions into pairs: [d0,d1], [d2,d3], ..., [d_{n-2},d_{n-1}]
    2. Each pair rotates in its 2D plane at frequency θ^(2i/d)
    3. Lower dims rotate faster (more position-sensitive)
    4. Higher dims rotate slower (more semantically stable)

    This base class provides:
    - Frequency computation
    - Dynamic sin/cos calculation
    - Precomputation utilities

    Subclasses implement specific application strategies:
    - PrecomputedRoPE: Precompute all positions for efficiency
    - PartialRoPE: Apply to subset of dimensions
    """

    def __init__(self, dim: int, config: position.RoPEConfig):
        """
        Initialize base RoPE.

        Args:
            dim: Dimension to apply RoPE to (must be even)
            config: RoPE configuration with theta
        """
        super().__init__()
        self.dim = dim
        self.config = config

        # Compute and store inverse frequencies
        inv_freq: jaxtyping.Float[torch.Tensor, "dim_half"]
        inv_freq = self._compute_inv_freq(dim, config.theta)
        # register_buffer vs nn.Parameter because:
        # inv_freq is precomputed and fixed - not learned during training
        # buffers are saved/loaded with the model but not updated by optimizer
        # buffers move with the model to the correct device automatically
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

        # Apply scaling if configured
        if self.config.scaling is not None:
            inv_freq = rope_scaling.apply_rope_scaling(inv_freq, self.config.scaling)

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

        # Apply scaling if configured
        if self.config.scaling is not None:
            inv_freq = rope_scaling.apply_rope_scaling(inv_freq, self.config.scaling)

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
