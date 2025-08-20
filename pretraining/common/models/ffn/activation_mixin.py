# Third Party
import torch
import torch.nn.functional as F
from torch import nn


class ReLUSquared(nn.Module):
    """
    ReLU-Squared activation function.
    Zhang et al., 2024, https://arxiv.org/pdf/2402.03804

    Computes F.relu(x).square() - provides ~1-2% improvement over GELU
    according to nanogpt experiments.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class ActivationMixin:
    """
    Mixin that provides activation function lookup by name.

    Centralizes activation function management for FFN modules,
    avoiding duplication across MLP, SwiGLU, and other FFN variants.
    """

    def _get_activation(self, activation: str) -> nn.Module:
        """
        Get activation function by name.

        Supports common activations used in transformer FFNs:
        - gelu: Smooth activation used by GPT-2, BERT
        - silu: Also known as swish, used by SwiGLU
        - relu: Classic activation, fast but less smooth
        - relu_squared: Experimental activation with slight improvements over GELU
        """
        activation_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),  # Also known as swish
            "relu_squared": ReLUSquared(),  # ~1-2% better than GELU per nanogpt
        }
        if activation not in activation_map:
            raise ValueError(
                f"Unknown activation: {activation}. Available: {list(activation_map.keys())}"
            )
        return activation_map[activation]
