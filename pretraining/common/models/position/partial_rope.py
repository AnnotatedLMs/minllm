# Standard Library

# Third Party
import jaxtyping
import torch

# Project
from pretraining.common.models.position import rope
from pretraining.common.models.position import yarn_mixins


class PartialRoPE(rope.RoPEBase, yarn_mixins.YaRNScalingMixin):
    """
    RoPE applied to only a subset of head dimensions with optional YaRN scaling.

    Originates from:
    - Su et al. RoFormer. https://arxiv.org/abs/2104.09864

    Used by: DeepSeek-V3's Multi-head Latent Attention (MLA)

    Variation: Applies RoPE to only rope_dim dimensions (e.g., 64 out of 128)
    Computation: Dynamic frequency calculation for position-only features
    Effect: Model learns to separate semantic content from positional information

    Supports YaRN scaling for context extension:
    - Pretraining: 4K context (no YaRN)
    - Phase 1: 4K → 32K (YaRN scale_factor = 8)
    - Phase 2: 32K → 128K (YaRN scale_factor = 32)
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
    ):
        """
        Initialize Partial RoPE.

        YaRN scaling is applied later via update_for_context_extension() during
        training phase transitions, not at initialization.

        Args:
            dim: RoPE dimension (e.g., 64 for DeepSeek-V3)
            theta: Base frequency
        """
        super().__init__(dim, theta=theta, linear_scaling=None)

        # YaRN state - will be set during context extension
        self.yarn_config = None
        self.yarn_mscale = 1.0

    def _initialize_yarn_scaling(self) -> None:
        """Initialize YaRN scaling based on configuration."""
        config = self.yarn_config

        # Apply YaRN scaling to frequencies
        scaled_inv_freq, mscale = self._apply_yarn_scaling(
            self.inv_freq,
            config.scale_factor,
            config.beta_fast,
            config.beta_slow,
            config.original_context_len,
            config.extrapolation_factor,
            config.attn_factor,
            config.mscale_all_dim,
        )

        # Update stored frequencies and mscale
        self.register_buffer("inv_freq", scaled_inv_freq, persistent=False)
        self.yarn_mscale = mscale

    def get_attention_scale_factor(self) -> float:
        """
        Get the attention scale factor including YaRN mscale adjustment.

        This should be used by the attention mechanism to adjust scaling
        when context is extended.
        """
        return self.yarn_mscale

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
        """Apply rotary positional embeddings to position-only vectors."""
        batch_size, num_heads, seq_len, rope_dim = x.shape

        # Compute sin/cos dynamically
        cos, sin = self._compute_dynamic_freqs(seq_len, x.device, x.dtype)

        # Expand cos/sin for all heads and batch
        cos = cos[None, None, :, :].expand(batch_size, num_heads, -1, -1)
        sin = sin[None, None, :, :].expand(batch_size, num_heads, -1, -1)

        # Apply rotation
        return self._apply_rotation(x, cos, sin, batch_size, num_heads, seq_len, rope_dim)
