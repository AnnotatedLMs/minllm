# Standard Library
import math
import typing

# Third Party
import jaxtyping
import torch


class YaRNScalingMixin:
    """
    Mixin for YaRN (Yet another RoPE extensioN) frequency scaling.

    Scholarship:
    - Peng et al., 2023, https://arxiv.org/pdf/2309.00071

    Significance:
    Enables context window extension while preserving both local and global position patterns.
    YaRN selectively scales different frequency bands - keeping high frequencies intact for local precision
    while interpolating low frequencies for extended range.

    Init:
    This mixin has no initialization. It provides utility methods for modules that contain RoPE:
        self.inv_freq: torch.Tensor  # Inverse frequencies buffer in the RoPE module
        self.theta: float  # Base frequency parameter (typically 10000)

    Step-by-step control flow (compute_yarn_scaling_for_context_extension):
    1. Calculate scale factor from new vs original context lengths
    2. Identify correction range using beta parameters to find frequency bands
    3. Create smooth ramp mask for transitioning between scaled/unscaled regions
    4. Apply interpolation to low frequencies, extrapolation to high frequencies
    5. Compute mscale factor for attention stability at extended lengths
    6. Return scaled frequencies and adjustment factors

    Learning process:
    - This mixin contains no learnable parameters.
    - Applies fixed mathematical transformations based on the YaRN algorithm.

    - Architectural benefit:
      - High frequency dimensions (small wavelengths) stay unscaled to preserve local token relationships
      - Low frequency dimensions (large wavelengths) get interpolated for extended context
      - Smooth transition via ramp function prevents abrupt frequency changes
      - mscale adjustment prevents attention entropy collapse at longer sequences
      - Result: Model can attend to extended contexts without losing fine-grained positional discrimination
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

    def compute_yarn_scaling_for_context_extension(
        self,
        current_inv_freq: torch.Tensor,
        new_max_seq_len: int,
        original_context_len: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        mscale_all_dim: float = 0.1,
    ) -> typing.Tuple[torch.Tensor, float, float]:
        """
        Compute YaRN-scaled frequencies for context extension.

        This method computes the scaled frequencies but doesn't update any buffers.
        The module using this mixin should handle buffer updates.

        This is called when transitioning between training phases:
        - Start of Phase 1: Update from 4K to 32K
        - Start of Phase 2: Update from 32K to 128K

        Args:
            current_inv_freq: Current inverse frequencies
            new_max_seq_len: New maximum sequence length
            original_context_len: Original pretraining context length
            beta_fast: YaRN beta_fast parameter (default: 32)
            beta_slow: YaRN beta_slow parameter (default: 1)
            extrapolation_factor: Extrapolation factor (default: 1)
            attn_factor: Attention scaling factor (default: 1)
            mscale_all_dim: mscale coefficient (default: 0.1)

        Returns:
            Tuple of (scaled_inv_freq, mscale, scale_factor)
        """
        # Calculate new scale factor
        scale_factor = new_max_seq_len / original_context_len

        # Apply YaRN scaling to frequencies
        scaled_inv_freq, mscale = self._apply_yarn_scaling(
            current_inv_freq,
            scale_factor,
            beta_fast,
            beta_slow,
            original_context_len,
            extrapolation_factor,
            attn_factor,
            mscale_all_dim,
        )

        return scaled_inv_freq, mscale, scale_factor
