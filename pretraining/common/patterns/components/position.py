# Standard Library
import math
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
# Local
from pretraining.configs.transformer import position_configs

# TODO: Explain buffer vs parameter distinction


class LearnedPositionEmbedding(nn.Module):
    """
    Learned positional embeddings (GPT-2 style).

    Simple wrapper around nn.Embedding for API consistency with RoPE.
    Each position gets its own learned embedding vector.
    """

    def __init__(
        self,
        max_position_embeddings: int,
        embedding_dim: int,
        init_std: float = 0.02,
    ):
        """
        Initialize learned position embeddings.

        Args:
            max_position_embeddings: Maximum number of positions
            embedding_dim: Dimension of embeddings (hidden_dim)
            init_std: Standard deviation for initialization
        """
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.embedding_dim = embedding_dim

        # Create embedding table
        self.wpe = nn.Embedding(max_position_embeddings, embedding_dim)

        # Initialize weights
        nn.init.normal_(self.wpe.weight, mean=0.0, std=init_std)

    def forward(
        self,
        position_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq embedding_dim"]:
        """
        Get position embeddings for given positions.

        Args:
            position_ids: Position indices [batch, seq]

        Returns:
            Position embeddings [batch, seq, embedding_dim]
        """
        if (position_ids >= self.max_position_embeddings).any():
            raise ValueError(
                f"Position ids must be less than max_position_embeddings "
                f"({self.max_position_embeddings})"
            )

        position_embeds: jaxtyping.Float[torch.Tensor, "batch seq embedding_dim"]
        position_embeds = self.wpe(position_ids)

        return position_embeds


class BaseRoPE(nn.Module):
    """
    Base class for Rotary Position Embeddings with shared functionality.

    Contains common operations:
    - Inverse frequency computation
    - Scaling for context extension
    - Precomputation utilities
    - Core rotation mathematics
    """

    def __init__(self, dim: int, config: position_configs.RoPEConfig):
        """
        Initialize base RoPE.

        Args:
            dim: Dimension to apply RoPE to
            config: RoPE configuration object
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
        """Compute inverse frequencies for RoPE."""
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        return inv_freq

    def _apply_scaling(
        self,
        freqs: jaxtyping.Float[torch.Tensor, "dim_half"],
    ) -> jaxtyping.Float[torch.Tensor, "dim_half"]:
        """
        Apply RoPE scaling for extended context length.

        Used by Llama 3.1 to extend from 8K -> 128K context.
        """
        if self.config.scaling is None:
            return freqs

        scaling = self.config.scaling
        low_freq_wavelen = scaling.original_context_len / scaling.low_freq_factor
        high_freq_wavelen = scaling.original_context_len / scaling.high_freq_factor

        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq

            if wavelen < high_freq_wavelen:
                # High frequency: no scaling
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                # Low frequency: full scaling
                new_freqs.append(freq / scaling.scale_factor)
            else:
                # Medium frequency: interpolated scaling
                smooth = (scaling.original_context_len / wavelen - scaling.low_freq_factor) / (
                    scaling.high_freq_factor - scaling.low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scaling.scale_factor + smooth * freq)

        scaled_freqs: jaxtyping.Float[torch.Tensor, "dim_half"]
        scaled_freqs = torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)
        return scaled_freqs

    def _precompute_freqs_cis(
        self,
        seq_len: int,
    ) -> jaxtyping.Float[torch.Tensor, "seq_len dim_half 2"]:
        """Precompute cos and sin values for all positions."""
        # Get inverse frequencies, potentially scaled
        freqs: jaxtyping.Float[torch.Tensor, "dim_half"] = self.inv_freq
        if self.config.scaling is not None:
            freqs = self._apply_scaling(freqs)

        # Create position indices
        positions: jaxtyping.Float[torch.Tensor, "seq_len"]
        positions = torch.arange(seq_len, dtype=torch.float32, device=freqs.device)

        # Compute outer product: position × frequency
        freqs_outer: jaxtyping.Float[torch.Tensor, "seq_len dim_half"]
        freqs_outer = torch.outer(positions, freqs)

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
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "seq_len dim_half"],
        jaxtyping.Float[torch.Tensor, "seq_len dim_half"],
    ]:
        """Compute sin/cos dynamically for the given sequence length."""
        # Get inverse frequencies on the right device
        freqs: jaxtyping.Float[torch.Tensor, "dim_half"] = self.inv_freq.to(device)
        if self.config.scaling is not None:
            freqs = self._apply_scaling(freqs)

        # Create positions
        positions: jaxtyping.Float[torch.Tensor, "seq_len"]
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)

        # Compute position × frequency
        sinusoid: jaxtyping.Float[torch.Tensor, "seq_len dim_half"]
        sinusoid = positions[:, None] * freqs[None, :]

        # Get sin and cos
        sin: jaxtyping.Float[torch.Tensor, "seq_len dim_half"] = torch.sin(sinusoid).to(dtype)
        cos: jaxtyping.Float[torch.Tensor, "seq_len dim_half"] = torch.cos(sinusoid).to(dtype)

        return cos, sin

    def get_freqs_cis(
        self,
        seq_len: int,
        device: torch.device,
    ) -> jaxtyping.Float[torch.Tensor, "seq_len dim_half 2"]:
        """
        Get precomputed or dynamically computed freqs_cis.

        This is useful for models that want to handle RoPE application themselves.
        """
        if (
            self.config.precompute
            and self.freqs_cis is not None
            and seq_len <= self.config.max_seq_len
        ):
            return self.freqs_cis[:seq_len].to(device)
        else:
            return self._precompute_freqs_cis(seq_len).to(device)


class PrecomputedRoPE(BaseRoPE):
    """
    RoPE implementation that precomputes frequencies for efficiency.

    Used by: Llama 3.1 and similar models

    Where this fits in the LLM:
    - Before: Q, K, V projections from attention layer
    - After: Scaled dot-product attention computation

    Order of operations for position encoding:
    1. At init: Precompute sin/cos for all positions up to max_seq_len
    2. At forward: Slice precomputed freqs for current seq_len
    3. Reshape input from [batch, seq, heads, head_dim] to [batch, seq, heads, head_dim//2, 2]
    4. Apply rotation via complex multiplication with sin/cos
    5. Reshape back to [batch, seq, heads, head_dim]

    Key features:
    - Applies to full head_dim
    - Trades memory for speed via precomputation
    - Supports RoPE scaling for extended context
    """

    def __init__(self, dim: int, config: position_configs.RoPEConfig, max_seq_len: int = 8192):
        """
        Initialize with precomputed frequencies.

        Args:
            dim: Dimension to apply RoPE to (typically head_dim)
            config: RoPE configuration with theta and optional scaling
            max_seq_len: Maximum sequence length to precompute for
        """
        super().__init__(dim, config)

        # Precompute and store frequencies for efficiency
        self.max_seq_len = max_seq_len
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(max_seq_len))

    def _apply_rotation(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq heads head_dim"],
        freqs_cis: jaxtyping.Float[torch.Tensor, "seq_len dim_half 2"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq heads head_dim"]:
        """Apply rotation using precomputed cos/sin values."""
        batch_size, seq_len, num_heads, head_dim = x.shape

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
        return self._apply_rotation(x, freqs_cis)


class PartialRoPE(BaseRoPE):
    """
    RoPE applied to only a subset of head dimensions.

    Used by: DeepSeek-V3's Multi-head Latent Attention (MLA)

    Where this fits in the LLM:
    - Before: Separate projections for keys_rope/queries_rope (rope_dim only)
    - After: Concatenation with content features before attention

    Order of operations for position encoding:
    1. Compute frequencies dynamically based on seq_len and rope_dim
    2. Apply scaling if configured (for extended context)
    3. Reshape input from [batch, heads, seq, rope_dim] to [batch, heads, seq, rope_dim//2, 2]
    4. Apply rotation via complex multiplication with sin/cos
    5. Reshape back to [batch, heads, seq, rope_dim]

    Key difference from PrecomputedRoPE:
    - Only applies to rope_dim dimensions (e.g., 64 out of 128 head_dim)
    - Computes frequencies on-the-fly rather than precomputing
    - Allows model to separate positional and content information
    """

    def _apply_rotation(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch heads seq rope_dim"],
        cos: jaxtyping.Float[torch.Tensor, "batch heads seq dim_half"],
        sin: jaxtyping.Float[torch.Tensor, "batch heads seq dim_half"],
    ) -> jaxtyping.Float[torch.Tensor, "batch heads seq rope_dim"]:
        """Apply rotation in DeepSeek format."""
        batch_size, num_heads, seq_len, rope_dim = x.shape

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
        return self._apply_rotation(x, cos, sin)
