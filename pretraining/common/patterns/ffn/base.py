# Standard Library
import typing

# Third Party
import torch.nn as nn

# Project
from pretraining.common.base import ffn


class FeedForward(ffn.BaseFeedForward):
    """
    Educational base class showing common feedforward patterns.

    Most feedforward networks share the same core operations:
    1. Project to higher dimension - increases capacity for learning complex transformations
    2. Apply non-linearity - enables learning non-linear relationships between features
    3. Project back to model dimension - compresses learned features back to original size
    4. Apply dropout - prevents overfitting by randomly dropping connections

    The variations between architectures typically involve:
    - How the intermediate dimension is calculated
    - Which activation functions are used
    - Whether gating mechanisms are employed
    - How information flows through the network

    See the forward() method in each implementation for the specific flow.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        dropout: typing.Optional[float] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.ffn_dropout = nn.Dropout(dropout) if dropout is not None else None

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),  # Also known as swish
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]
