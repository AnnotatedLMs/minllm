# Standard Library
import math

# Third Party
import torch
import torch.nn as nn


class TransformerWeightInitializer:
    """
    Weight initializer for transformer models following GPT-2 scheme.

    The GPT-2 paper introduced a specific initialization strategy for deep
    transformers that helps maintain stable gradients across many layers.
    This is critical for training stability and convergence.
    """

    def __init__(
        self,
        n_layer: int,
        std: float = 0.02,
        residual_pattern: str = "c_proj.weight",
    ):
        """
        Initialize the weight initializer.

        Args:
            n_layer: Number of transformer layers in the model
            std: Base standard deviation for weight initialization
            residual_pattern: String pattern to identify residual projections
        """
        self.n_layer = n_layer
        self.std = std
        self.residual_pattern = residual_pattern

        # Calculate scaled std for residual projections
        self.residual_std = self._calculate_residual_std()

    def _calculate_residual_std(self) -> float:
        """
        Calculate the scaled standard deviation for residual projections.

        The scaling factor 1/sqrt(2*n_layer) accounts for the variance
        accumulation through residual connections. Each layer adds variance,
        so we scale down proportionally to the number of layers.

        Returns:
            Scaled standard deviation for residual projections
        """
        return self.std / math.sqrt(2 * self.n_layer)

    def _init_linear(self, module: nn.Linear) -> None:
        """
        Initialize a linear layer.

        Uses normal distribution for weights and zeros for bias.

        Args:
            module: Linear layer to initialize
        """
        torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def _init_embedding(self, module: nn.Embedding) -> None:
        """
        Initialize an embedding layer.

        Uses normal distribution matching the linear layer initialization.

        Args:
            module: Embedding layer to initialize
        """
        torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)

    def _init_standard_module(self, module: nn.Module) -> None:
        """
        Apply standard initialization to a module.

        This is the first pass that initializes all weights uniformly.

        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            self._init_linear(module)
        elif isinstance(module, nn.Embedding):
            self._init_embedding(module)
        elif isinstance(module, nn.LayerNorm):
            self._init_layernorm(module)

    def _init_layernorm(self, module: nn.LayerNorm) -> None:
        """
        Initialize a LayerNorm layer.

        Standard approach: weights to 1, bias to 0.

        Args:
            module: LayerNorm layer to initialize
        """
        torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def _apply_standard_init(self, model: nn.Module) -> None:
        """
        Apply standard initialization to all modules in the model.

        This is the first initialization pass.

        Args:
            model: The full model to initialize
        """
        model.apply(self._init_standard_module)

    def _apply_residual_scaling(self, model: nn.Module) -> None:
        """
        Apply special scaling to residual projection layers.

        This is the second pass that identifies residual projections
        (like attention output and FFN output) and re-initializes them
        with the scaled standard deviation.

        Args:
            model: The full model containing residual projections
        """
        for param_name, param in model.named_parameters():
            if param_name.endswith(self.residual_pattern):
                torch.nn.init.normal_(param, mean=0.0, std=self.residual_std)

    def initialize(self, model: nn.Module) -> None:
        """
        Initialize all weights in the transformer model.

        This is the main entry point that coordinates the two-phase
        initialization process:
        1. Standard initialization for all weights
        2. Scaled initialization for residual projections

        Args:
            model: The transformer model to initialize
        """
        # Phase 1: Initialize all weights with standard approach
        self._apply_standard_init(model)

        # Phase 2: Re-initialize residual projections with scaling
        self._apply_residual_scaling(model)
