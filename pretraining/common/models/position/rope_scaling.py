# Standard Library
import math

# Third Party
import jaxtyping
import torch

# Project
from pretraining.configs.model.components import position


def apply_rope_scaling(
    freqs: jaxtyping.Float[torch.Tensor, "dim_half"],
    scaling_config: position.LinearRoPEScalingConfig,
) -> jaxtyping.Float[torch.Tensor, "dim_half"]:
    """
    Apply RoPE scaling for extended context length.

    This is the key to Llama 3.1's context extension from 8K to 128K!
    NOT a separate RoPE variant - just modifies the frequencies.

    The algorithm divides frequencies into three zones:
    1. High freq (local patterns): No scaling - preserves fine-grained position info
       These handle nearby token relationships (e.g., adjacent words)
    2. Low freq (global patterns): Scale by factor - extends long-range dependencies
       These handle document-level patterns (e.g., intro vs conclusion)
    3. Medium freq: Smooth interpolation - gradual transition between zones

    Example with scale_factor=8 (8K→64K):
    - High freq dimensions: Unchanged, still distinguish positions 0,1,2...
    - Low freq dimensions: Scaled 8x slower, now one "cycle" covers 8x more tokens
    - Medium freq: Gradually transitions from no scaling to full scaling

    This allows models trained on shorter contexts to work on much longer
    sequences while maintaining most learned position-based behaviors.

    Args:
        freqs: Original inverse frequencies from RoPE
        scaling_config: Configuration with scale_factor and frequency boundaries

    Returns:
        Scaled frequencies that extend the model's context length
    """
    low_freq_wavelen = scaling_config.original_context_len / scaling_config.low_freq_factor
    high_freq_wavelen = scaling_config.original_context_len / scaling_config.high_freq_factor

    new_freqs = []
    for freq in freqs:
        # Convert frequency to wavelength (how many positions for one full rotation)
        wavelen = 2 * math.pi / freq

        if wavelen < high_freq_wavelen:
            # High frequency: no scaling
            # These dimensions rotate quickly, handling local patterns
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            # Low frequency: full scaling
            # These dimensions rotate slowly, handling global patterns
            new_freqs.append(freq / scaling_config.scale_factor)
        else:
            # Medium frequency: interpolated scaling
            # Smooth transition between high and low frequency zones
            smooth = (
                scaling_config.original_context_len / wavelen - scaling_config.low_freq_factor
            ) / (scaling_config.high_freq_factor - scaling_config.low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scaling_config.scale_factor + smooth * freq)

    scaled_freqs: jaxtyping.Float[torch.Tensor, "dim_half"]
    scaled_freqs = torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)
    return scaled_freqs
