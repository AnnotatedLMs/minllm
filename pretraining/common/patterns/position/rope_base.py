# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from pretraining.common.patterns.position import rope_scaling
from pretraining.configs.model.components import position


class BaseRoPE(nn.Module):
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
    - StandardRoPE: Apply to full head dimension
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
