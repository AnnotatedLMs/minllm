# Third Party
import jaxtyping
import torch


class PositionIDGenerationMixin:
    """
    Mixin for generating position IDs for embeddings.

    Variation: Standard sequential position IDs
    Computation: Creates position tensor [0, 1, 2, ...] and expands for batch
    Effect: Provides position indices for position embeddings
    """

    def _get_position_ids(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> jaxtyping.Int[torch.Tensor, "batch seq"]:
        """Generate standard position IDs [0, 1, 2, ...]."""
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        return position_ids.unsqueeze(0).expand(batch_size, -1)
