# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from pretraining.gpt.models import config


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: config.GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def _expand_hidden_dimension(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq 4*d_model"]:
        """Expand to 4x hidden dimension."""
        return self.c_fc(x)

    def _apply_activation(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq 4*d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq 4*d_model"]:
        """Apply GELU activation function."""
        return self.gelu(x)

    def _project_back_to_model_dimension(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq 4*d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Project back to original model dimension."""
        return self.c_proj(x)

    def _apply_dropout(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """Apply dropout for regularization."""
        return self.dropout(x)

    def forward(
        self, x: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply feed-forward network.

        The process:
        1. Expand to 4x hidden dimension
        2. Apply GELU activation
        3. Project back to model dimension
        4. Apply dropout
        """
        x = self._expand_hidden_dimension(x)
        x = self._apply_activation(x)
        x = self._project_back_to_model_dimension(x)
        x = self._apply_dropout(x)
        return x
