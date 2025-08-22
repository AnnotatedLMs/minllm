# Standard Library
import typing

# Third Party
import torch
from torch import nn


class CacheManagementMixin:
    """
    Mixin for managing KV caches across transformer blocks.

    Variation: Coordinates cache lifecycle at architecture level
    Computation: Install/clear caches in all attention layers
    Effect: Enables efficient generation with consistent memory management

    Used by: Llama3, any architecture with cached attention layers
    """

    def install_kv_caches(
        self,
        blocks: nn.ModuleList,
        batch_size: int,
        max_seq_length: int,
        dtype: torch.dtype = torch.float16,
        device: typing.Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Install KV caches in all attention layers of provided blocks.

        Args:
            blocks: Transformer blocks containing attention modules
            batch_size: Maximum batch size to support
            max_seq_length: Maximum sequence length to cache
            dtype: Data type for cache tensors
            device: Device to allocate cache on
        """
        for block in blocks:
            if hasattr(block, "attention") and hasattr(block.attention, "setup_cache"):
                # Attention module should have num_kv_heads and head_dim attributes
                block.attention.setup_cache(
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    n_kv_heads=block.attention.num_kv_heads,
                    head_dim=block.attention.head_dim,
                    dtype=dtype,
                    device=device,
                )

    def clear_kv_caches(self, blocks: nn.ModuleList) -> None:
        """
        Clear KV caches from all attention layers in provided blocks.

        Args:
            blocks: Transformer blocks containing attention modules
        """
        for block in blocks:
            if hasattr(block, "attention") and hasattr(block.attention, "reset_cache"):
                block.attention.reset_cache()
