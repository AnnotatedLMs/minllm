# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn


class KVCache(nn.Module):
    """
    Key-Value cache for efficient autoregressive generation.

    Used by: Llama-style models for inference optimization

    Core operations:
    1. Pre-allocate static buffers - reserves GPU memory for maximum sequence length
    2. Update cache incrementally - stores new K/V states at the current position
    3. Return accumulated states - provides full history for attention computation

    The cache enables O(1) token generation instead of O(n) by avoiding recomputation
    of past key/value states during autoregressive decoding.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_length: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: typing.Union[torch.device, str] = "cuda",
    ):
        """
        Initialize KV cache with pre-allocated buffers.

        Pre-allocation prevents dynamic memory allocation during generation,
        ensuring consistent latency and avoiding OOM errors.
        """
        super().__init__()

        cache_shape = (batch_size, max_seq_length, n_kv_heads, head_dim)
        # Buffers: store intermediate K/V activations across generation steps
        # Not parameters since these are computed activations, not learned weights
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

        # Buffer: tracks current position in cache for sequential generation
        self.register_buffer("cache_position", torch.zeros(1, dtype=torch.long, device=device))

    def update(
        self,
        start_pos: int,
        xk: jaxtyping.Float[torch.Tensor, "batch seq_len n_kv_heads head_dim"],
        xv: jaxtyping.Float[torch.Tensor, "batch seq_len n_kv_heads head_dim"],
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch cached_seq n_kv_heads head_dim"],
        jaxtyping.Float[torch.Tensor, "batch cached_seq n_kv_heads head_dim"],
    ]:
        """
        Update cache with new key-value pairs and return full cached sequences.

        The process:
        1. Insert new K/V states at start_pos - enables incremental generation
        2. Return full cached sequence - provides complete history for attention

        This avoids recomputing K/V for all previous tokens on each generation step.
        """
        seq_len = xk.size(1)

        # Update the cache with new key-value pairs
        self.cache_k[:, start_pos : start_pos + seq_len] = xk
        self.cache_v[:, start_pos : start_pos + seq_len] = xv

        # Return the cached sequences up to the current position
        cached_k = self.cache_k[:, : start_pos + seq_len]
        cached_v = self.cache_v[:, : start_pos + seq_len]

        return cached_k, cached_v

    def reset(self) -> None:
        """Reset the cache by zeroing out all stored keys and values - prepares for new sequence."""
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.cache_position.zero_()
