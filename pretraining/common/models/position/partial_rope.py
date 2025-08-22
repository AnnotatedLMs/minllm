# Standard Library

# Third Party
import jaxtyping
import torch

# Project
from pretraining.common.models.position import rope
from pretraining.common.models.position import yarn_mixins


class DecoupledRoPE(rope.RoPEBase, yarn_mixins.YaRNScalingMixin):
    """
    Applies rotary position embeddings to position-only features in decoupled RoPE strategy.

    Scholarship:
    - Su et al. RoFormer, 2021. https://arxiv.org/abs/2104.09864
    - Deepseek-V2, 2024, https://arxiv.org/pdf/2405.04434, Section 2.1.3

    Significance:
    Enables aggressive KV compression by separating position from content processing.
    Position features get rotary embeddings while compressed content stays untouched.
    This decoupling allows the model to cache position and content separately.

    Init:
    Inherits from RoPEBase which registers inverse frequencies:
        self.register_buffer("inv_freq", inv_freq)  # Mathematical constants for rotation
    YaRN mixin provides scaling methods but doesn't initialize anything.

    Step-by-step control flow (forward):
    1. Receive position features of shape [batch, heads, seq, rope_dim]
    2. Compute sin/cos values dynamically based on sequence positions
    3. Expand sin/cos for all heads and batch dimensions
    4. Apply 2D rotation to each dimension pair
    5. Return rotated position features

    Learning process:
    - This module contains no learnable parameters.
    - The inverse frequencies are fixed mathematical constants based on theta and dimensions.
    - Rotations are deterministic transformations that encode absolute position.

    - Architectural benefit:
      - Position features flow through separate gradient paths from content
      - Allows content compression (W_DKV) to focus purely on semantic information
      - Position-only rotations preserve fine-grained location awareness
      - During inference, rotated positions can be cached independently
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
        self.yarn_mscale = 1.0
        self.yarn_scale_factor = 1.0
        self.yarn_original_context_len = None

    def update_for_context_extension(
        self,
        new_max_seq_len: int,
        original_context_len: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        mscale_all_dim: float = 0.1,
    ) -> None:
        """
        Update RoPE frequencies for context extension training phase.

        This method is called when transitioning between training phases:
        - Start of Phase 1: Update from 4K to 32K
        - Start of Phase 2: Update from 32K to 128K

        Args:
            new_max_seq_len: New maximum sequence length
            original_context_len: Original pretraining context length
            beta_fast: YaRN beta_fast parameter (default: 32)
            beta_slow: YaRN beta_slow parameter (default: 1)
            extrapolation_factor: Extrapolation factor (default: 1)
            attn_factor: Attention scaling factor (default: 1)
            mscale_all_dim: mscale coefficient (default: 0.1)
        """
        # Use mixin to compute scaled frequencies
        scaled_inv_freq, mscale, scale_factor = self.compute_yarn_scaling_for_context_extension(
            # buffer defined in RoPEBase
            self.inv_freq,
            new_max_seq_len,
            original_context_len,
            beta_fast,
            beta_slow,
            extrapolation_factor,
            attn_factor,
            mscale_all_dim,
        )

        # Update our buffer with the scaled frequencies
        self.register_buffer("inv_freq", scaled_inv_freq, persistent=False)
        self.yarn_mscale = mscale
        self.yarn_scale_factor = scale_factor
        self.yarn_original_context_len = original_context_len

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
