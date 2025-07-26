# Third Party
import jaxtyping
import torch
import torch.nn as nn


class MultiTokenPredictionHead(nn.Module):
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
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.depth = depth

        self.proj_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch depth seq vocab_size"]:
        """
        Predict multiple future tokens at each position.

        The process:
        1. Project hidden states - transforms representations for each future position
        2. Apply dropout - regularizes to prevent overfitting on auxiliary task
        3. Normalize - stabilizes predictions across different depths
        4. Generate logits - produces vocabulary distribution for each future position
        5. Stack predictions - organizes outputs by prediction depth

        Each depth uses the previous depth's projections as input, creating
        a cascade that specializes in increasingly distant predictions.
        """
        predictions = []
        current_hidden = hidden_states

        for d in range(self.depth):
            # Project the hidden states
            projected = self.proj_layers[d](current_hidden)

            # Apply dropout
            projected = self.dropout(projected)

            # Normalize
            normalized = self.norm(projected)

            # Get logits for this depth
            logits = self.output_head(normalized)
            predictions.append(logits)

            # Use projected states as input for next depth
            current_hidden = projected

        # Stack predictions along depth dimension
        return torch.stack(predictions, dim=1)
