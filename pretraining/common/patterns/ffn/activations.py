# Third Party
import torch
import torch.nn.functional as F
from torch import nn


class ReLUSquared(nn.Module):
    """
    ReLU-Squared activation function.

    Computes F.relu(x).square() - provides ~1-2% improvement over GELU
    according to nanogpt experiments.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()
