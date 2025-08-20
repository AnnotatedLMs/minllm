# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn


class TransformerBlockStackMixin:
    """
    Mixin for managing transformer block iteration.

    Variation: Sequential application of transformer blocks
    Computation: Pass hidden states through each block in order
    Effect: Builds up representations through depth
    """

    def _apply_transformer_blocks(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
        blocks: nn.ModuleList,
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_offset: int = 0,
        output_hidden_states: bool = False,
        final_norm: typing.Optional[nn.Module] = None,
        **kwargs,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
        typing.Optional[typing.List[jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"]]],
    ]:
        """
        Pass input through all transformer blocks.

        Args:
            x: Input hidden states
            blocks: ModuleList of transformer blocks
            attention_mask: Optional padding mask
            position_offset: Offset for position embeddings (KV caching)
            output_hidden_states: Whether to return all intermediate states
            final_norm: Optional final normalization layer
            **kwargs: Additional arguments for blocks

        Returns:
            Tuple of (final_hidden_states, all_hidden_states_if_requested)
        """
        all_hidden_states = [] if output_hidden_states else None

        hidden = x
        for block in blocks:
            if output_hidden_states:
                all_hidden_states.append(hidden)

            # Pass through transformer block
            hidden = block(
                hidden, attention_mask=attention_mask, position_offset=position_offset, **kwargs
            )

        # Apply final normalization if provided
        if final_norm is not None:
            hidden = final_norm(hidden)

        if output_hidden_states:
            all_hidden_states.append(hidden)

        return hidden, all_hidden_states


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
