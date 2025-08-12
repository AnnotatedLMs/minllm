# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.base import core


class MultiTokenPredictionHead(core.BaseTorchModule):
    """
    Multi-token prediction head for predicting multiple future tokens.

    Used by: DeepSeek-V3 (auxiliary head)

    Variation: Predicts multiple future tokens (n+1, n+2, ..., n+depth) at each position
    Computation: Separate projection layers for each future position
    Effect: Model learns longer-range dependencies and improves sample efficiency

    The multi-token prediction acts as an auxiliary task during training,
    helping the model develop better representations of future context.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        depth: int = 1,
        dropout: typing.Optional[float] = None,
    ):
        super().__init__()
        self.depth = depth

        self.proj_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.mtp_dropout = nn.Dropout(dropout) if dropout is not None else None
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> typing.List[jaxtyping.Float[torch.Tensor, "batch seq_len vocab_size"]]:
        """
        Predict multiple future tokens at each position.

        The process:
        1. Project hidden states - transforms representations for each future position
        2. Apply dropout - regularizes to prevent overfitting on auxiliary task
        3. Normalize - stabilizes predictions across different depths
        4. Generate logits - produces vocabulary distribution for each future position
        5. Return list of predictions for each depth

        Each depth uses the previous depth's projections as input, creating
        a cascade that specializes in increasingly distant predictions.

        Returns:
            List of logits tensors, one for each future position
        """
        predictions = []
        current_hidden = hidden_states

        for d in range(self.depth):
            # Project the hidden states
            projected = self.proj_layers[d](current_hidden)

            # Apply dropout
            projected = self._maybe_apply_dropout(projected, self.mtp_dropout)

            # Normalize
            normalized = self.norm(projected)

            # Get logits for this depth
            logits = self.output_head(normalized)
            predictions.append(logits)

            # Use projected states as input for next depth
            current_hidden = projected

        # Return list of predictions
        return predictions
