# Standard Library
import typing

# Third Party
import jaxtyping
import torch


class CachedAttentionMixin:
    """
    Mixin that adds KV cache capabilities to attention modules.

    Scholarship:
    Efficiently Scaling Transformer Inference, 2022, https://arxiv.org/pdf/2211.05102

    Significance:
    Enables O(1) memory per token during autoregressive generation instead of O(n).
    Without caching, we'd recompute all past K/V states every generation step.
    Static pre-allocation avoids dynamic memory allocation and OOM errors during inference.

    Init:
    This mixin has no initialization. It dynamically adds buffers when setup_cache is called:
        self.cache_k = torch.zeros(cache_shape)  # Stores accumulated key states
        self.cache_v = torch.zeros(cache_shape)  # Stores accumulated value states
        self.cache_position = torch.zeros(1)     # Tracks current position in cache

    Step-by-step control flow (setup_cache):
    1. Pre-allocate static buffers for maximum sequence length
    2. Register K and V cache tensors as buffers (not parameters)
    3. Initialize position tracker to zero

    Step-by-step control flow (update_kv_cache_with_transpose):
    1. Transpose K/V from attention format [batch, heads, seq, dim] to cache format
    2. Insert new K/V states at current position in cache
    3. Return accumulated K/V up to current position
    4. Transpose back to attention format for computation

    Learning process:
    - This mixin contains no learnable parameters
    - Cache buffers store computed activations, not learned weights

    - Operational benefits:
      - Avoids recomputing K/V for all previous tokens on each generation step
      - Reduces memory bandwidth: only load new K/V instead of full sequence
      - Enables efficient incremental generation for long sequences
      - Pre-allocation prevents latency spikes from dynamic memory allocation
    """

    def setup_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: typing.Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Initialize KV cache with pre-allocated buffers.

        Pre-allocation prevents dynamic memory allocation during generation,
        ensuring consistent latency and avoiding OOM errors.
        """
        cache_shape = (batch_size, max_seq_length, n_kv_heads, head_dim)

        # Buffers: store intermediate K/V activations across generation steps
        # Not parameters since these are computed activations, not learned weights
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

        # Buffer: tracks current position in cache for sequential generation
        self.register_buffer("cache_position", torch.zeros(1, dtype=torch.long, device=device))

    def update_kv_cache_with_transpose(
        self,
        k_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"],
        v_heads: jaxtyping.Float[torch.Tensor, "batch n_heads seq head_dim"],
        position_offset: int,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch n_heads cached_seq head_dim"],
        jaxtyping.Float[torch.Tensor, "batch n_heads cached_seq head_dim"],
    ]:
        """
        Update KV cache handling the transpose from attention format to cache format and back.

        Why the transpose is needed:
        - Attention computation wants [batch, n_heads, seq, head_dim] because matrix multiplication
          happens between Q and K^T where we need seq dimensions to be the last two dims for matmul
        - Cache storage wants [batch, seq, n_heads, head_dim] because we update the cache by
          position (seq dimension), so having seq as the second dimension allows efficient slicing
          like cache[:, position:position+new_len] = new_values

        This method handles both transposes and the cache update as a single operation.
        """
        # Transpose from attention format to cache format
        # [batch, n_heads, seq, head_dim] -> [batch, seq, n_heads, head_dim]
        k_for_cache = k_heads.transpose(1, 2)
        v_for_cache = v_heads.transpose(1, 2)

        # Update cache and get back accumulated keys/values
        k_cached, v_cached = self.update_and_get_cached_kv(
            position_offset, k_for_cache, v_for_cache
        )

        # Transpose back to attention format
        # [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
        k_heads_cached = k_cached.transpose(1, 2)
        v_heads_cached = v_cached.transpose(1, 2)

        return k_heads_cached, v_heads_cached

    def update_and_get_cached_kv(
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
        if not hasattr(self, "cache_k"):
            # No cache setup, return as-is (normal forward without caching)
            return xk, xv

        seq_len = xk.size(1)

        # Update the cache with new key-value pairs
        self.cache_k[:, start_pos : start_pos + seq_len] = xk
        self.cache_v[:, start_pos : start_pos + seq_len] = xv

        # Return the cached sequences up to the current position
        cached_k = self.cache_k[:, : start_pos + seq_len]
        cached_v = self.cache_v[:, : start_pos + seq_len]

        return cached_k, cached_v

    def reset_cache(self) -> None:
        """Reset the cache by zeroing out all stored keys and values - prepares for new sequence."""
        if hasattr(self, "cache_k"):
            self.cache_k.zero_()
            self.cache_v.zero_()
            self.cache_position.zero_()

    @property
    def has_cache(self) -> bool:
        """Check if cache is currently active."""
        return hasattr(self, "cache_k")
