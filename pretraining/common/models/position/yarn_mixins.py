# Standard Library
import math
import typing

# Third Party
import jaxtyping
import torch


class YaRNScalingMixin:
    """
    Mixin for YaRN scaling.
    Peng et al., 2023, https://arxiv.org/pdf/2309.00071

    Variation: Dynamic context extension with beta-based correction
    Computation: Interpolates frequencies with correction range
    Effect: Extends context length while preserving local and global patterns

    Used by: DeepSeek-V3 during context extension training phases

    The YaRN algorithm differs from standard RoPE scaling:
    1. Uses beta_fast/beta_slow to determine correction range
    2. Applies extrapolation factor for dimensions outside range
    3. Includes mscale adjustment for attention stability

    Context extension phases (DeepSeek-V3):
    - Pretraining: 4K context
    - Phase 1: 4K → 32K (scale_factor = 8)
    - Phase 2: 32K → 128K (scale_factor = 32)
    """

    def _find_yarn_correction_dim(
        self,
        num_rotations: float,
        dim: int,
        base: float,
        max_position_embeddings: int,
    ) -> float:
        """
        Find dimension index for a given number of rotations.

        This inverts the frequency formula to find which dimension
        corresponds to a specific rotation count.
        """
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def _find_yarn_correction_range(
        self,
        beta_fast: float,
        beta_slow: float,
        dim: int,
        base: float,
        original_context_len: int,
    ) -> typing.Tuple[int, int]:
        """
        Find correction range using YaRN's beta parameters.

        Beta parameters control which frequency bands get scaled:
        - beta_fast: High frequency cutoff (e.g., 32)
        - beta_slow: Low frequency cutoff (e.g., 1)
        """
        low = math.floor(self._find_yarn_correction_dim(beta_fast, dim, base, original_context_len))
        high = math.ceil(self._find_yarn_correction_dim(beta_slow, dim, base, original_context_len))
        return max(low, 0), min(high, dim - 1)

    def _create_yarn_ramp_mask(
        self,
        low: float,
        high: float,
        dim: int,
    ) -> jaxtyping.Float[torch.Tensor, "dim"]:
        """
        Create linear ramp for smooth interpolation between scaled/unscaled regions.

        Returns values from 0 to 1, where:
        - 0 means full scaling (interpolation)
        - 1 means no scaling (extrapolation)
        """
        if low == high:
            high += 0.001  # Prevent division by zero

        linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def _compute_yarn_mscale(
        self,
        scale_factor: float,
        mscale_all_dim: float = 0.1,
    ) -> float:
        """
        Compute mscale factor for attention stability.

        The mscale factor adjusts attention scaling when context is extended,
        preventing attention entropy collapse at longer sequences.

        Formula: mscale = 0.1 * ln(scale_factor) + 1.0
        """
        if scale_factor <= 1:
            return 1.0
        return mscale_all_dim * math.log(scale_factor) + 1.0

    def _apply_yarn_scaling(
        self,
        inv_freq: jaxtyping.Float[torch.Tensor, "dim_half"],
        scale_factor: float,
        beta_fast: float,
        beta_slow: float,
        original_context_len: int,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        mscale_all_dim: float = 0.1,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "dim_half"],
        float,  # mscale
    ]:
        """
        Apply YaRN scaling to RoPE frequencies.

        Args:
            inv_freq: Original inverse frequencies
            scale_factor: Context extension factor (e.g., 8 for 4K→32K)
            beta_fast: High frequency cutoff
            beta_slow: Low frequency cutoff
            original_context_len: Original training context length
            extrapolation_factor: Factor for extrapolated dimensions
            attn_factor: Additional attention scaling factor
            mscale_all_dim: Coefficient for mscale computation

        Returns:
            Tuple of (scaled_frequencies, mscale_factor)
        """
        # Compute mscale for attention adjustment
        mscale = self._compute_yarn_mscale(scale_factor, mscale_all_dim)
        if attn_factor != 1.0:
            mscale *= attn_factor

        # If no scaling needed, return original
        if scale_factor <= 1:
            return inv_freq, mscale

        # Convert inverse frequencies to frequencies
        dim = len(inv_freq) * 2
        theta = self.theta if hasattr(self, "theta") else 10000.0
        pos_freqs = theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)

        # Prepare interpolation and extrapolation frequencies
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale_factor * pos_freqs)

        # Find correction range using beta parameters
        low, high = self._find_yarn_correction_range(
            beta_fast, beta_slow, dim, theta, original_context_len
        )

        # Create smooth transition mask
        inv_freq_mask = (
            1 - self._create_yarn_ramp_mask(low, high, dim // 2).to(inv_freq.device)
        ) * extrapolation_factor

        # Apply YaRN scaling: interpolate most dims, extrapolate high-freq dims
        scaled_inv_freq = (
            inv_freq_interpolation.to(inv_freq.device) * (1 - inv_freq_mask)
            + inv_freq_extrapolation.to(inv_freq.device) * inv_freq_mask
        )

        return scaled_inv_freq.to(inv_freq.dtype), mscale

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
        # Calculate new scale factor
        scale_factor = new_max_seq_len / original_context_len

        # Apply YaRN scaling to frequencies
        scaled_inv_freq, mscale = self._apply_yarn_scaling(
            self.inv_freq,
            scale_factor,
            beta_fast,
            beta_slow,
            original_context_len,
            extrapolation_factor,
            attn_factor,
            mscale_all_dim,
        )

        # Update stored frequencies and mscale
        self.register_buffer("inv_freq", scaled_inv_freq, persistent=False)
        self.yarn_mscale = mscale

        # Store current configuration
        self.yarn_scale_factor = scale_factor
        self.yarn_original_context_len = original_context_len
