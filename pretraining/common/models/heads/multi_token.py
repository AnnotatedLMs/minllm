# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn


class MultiTokenPredictionHead(nn.Module):
    """
    Multi-token prediction head for predicting multiple future tokens.

    Comes from:
    Xia et al., Speculative Decoding, (2023) https://aclanthology.org/2023.findings-emnlp.257/
    Leviathan et al., Speculative Decoding, (2023) https://proceedings.mlr.press/v202/leviathan23a.html

    Gloeckle et al., MTP, (2024) https://openreview.net/forum?id=pEWAcejiU2
    Li et al., EAGLE, (2024) https://openreview.net/forum?id=1NdN7eXyb4

    Used by: DeepSeek-V3 (auxiliary head)

    Variation: Predicts multiple future tokens (n+1, n+2, ..., n+depth) at each position
    Computation: Separate projection layers for each future position
    Effect: Model learns longer-range dependencies and improves sample efficiency
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        depth: int = 3,
        dropout: typing.Optional[float] = None,
    ):
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Projection layers for each future position
        self.proj_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])

        # Normalization for stability
        self.norm = nn.LayerNorm(hidden_dim)

        # Dropout for regularization
        self.mtp_dropout = nn.Dropout(dropout) if dropout is not None else None

        # Output head (can be shared across depths)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> typing.List[jaxtyping.Float[torch.Tensor, "batch seq_len vocab_size"]]:
        """Predict multiple future tokens at each position."""
        predictions = []
        current_hidden = hidden_states

        for d in range(self.depth):
            # Project the hidden states
            projected: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
            projected = self.proj_layers[d](current_hidden)

            # Apply dropout if configured
            if self.mtp_dropout is not None and self.training:
                projected = self.mtp_dropout(projected)

            # Normalize
            normalized: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
            normalized = self.norm(projected)

            # Get logits for this depth
            logits: jaxtyping.Float[torch.Tensor, "batch seq_len vocab_size"]
            logits = self.output_head(normalized)
            predictions.append(logits)

            # Use projected states as input for next depth
            current_hidden = projected

        return predictions
