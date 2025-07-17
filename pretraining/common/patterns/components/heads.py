# Third Party
import jaxtyping
import torch
import torch.nn as nn

# TODO: Needs review


class MultiTokenPredictionHead(nn.Module):
    """
    Multi-token prediction head for predicting multiple future tokens.

    Used by: DeepSeek-V3 (auxiliary head)

    Based on the DeepSeek training script implementation.
    Predicts n+1, n+2, ..., n+depth tokens at each position.
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

        Args:
            hidden_states: Final transformer hidden states [batch, seq, hidden_dim]

        Returns:
            Logits for multiple future tokens [batch, depth, seq, vocab_size]
            where depth=0 predicts next token, depth=1 predicts token after next, etc.
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
