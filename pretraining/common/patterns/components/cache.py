# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn


class KVCache(nn.Module):
    """
    Key-Value cache for efficient autoregressive generation.

    Pre-allocates static buffers to store past key and value states,
    avoiding recomputation during generation.

    Used by Llama-style models for inference optimization.
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

        Args:
            batch_size: Maximum batch size to support
            max_seq_length: Maximum sequence length to cache
            n_kv_heads: Number of key-value heads (for GQA)
            head_dim: Dimension of each head
            dtype: Data type for cache tensors
            device: Device to allocate cache on
        """
        super().__init__()

        cache_shape = (batch_size, max_seq_length, n_kv_heads, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

        # Track the current position in the cache
        self.register_buffer("cache_position", torch.zeros(1, dtype=torch.long, device=device))

    def update(
        self,
        start_pos: int,
        xk: jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads head_dim"],
        xv: jaxtyping.Float[torch.Tensor, "batch seq n_kv_heads head_dim"],
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch cached_seq n_kv_heads head_dim"],
        jaxtyping.Float[torch.Tensor, "batch cached_seq n_kv_heads head_dim"],
    ]:
        """
        Update cache with new key-value pairs and return full cached sequences.

        Args:
            start_pos: Position to insert new KV states
            xk: New key states
            xv: New value states

        Returns:
            Tuple of (cached_keys, cached_values) including history
        """
        seqlen = xk.size(1)

        # Update the cache with new key-value pairs
        self.cache_k[:, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, start_pos : start_pos + seqlen] = xv

        # Return the cached sequences up to the current position
        cached_k = self.cache_k[:, : start_pos + seqlen]
        cached_v = self.cache_v[:, : start_pos + seqlen]

        return cached_k, cached_v

    def reset(self) -> None:
        """Reset the cache by zeroing out all stored keys and values."""
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.cache_position.zero_()
