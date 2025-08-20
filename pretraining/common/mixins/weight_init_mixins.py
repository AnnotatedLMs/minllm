# Standard Library
import math
import typing

# Third Party
import torch
from torch import nn


class TransformerWeightInitMixin:
    """
    Mixin for transformer weight initialization patterns.

    Implements GPT-2 style two-phase initialization:
    1. Standard initialization for all weights
    2. Scaled initialization for residual projections

    The scaling factor 1/sqrt(2*n_layers) prevents variance explosion
    across deep networks with residual connections.
    """

    def _apply_weight_initialization(
        self,
        n_layers: int,
        std: float = 0.02,
        residual_pattern: str = "c_proj.weight",
        position_embeddings: typing.Optional[nn.Module] = None,
        position_init_std: float = 0.02,
    ) -> None:
        """
        Apply transformer weight initialization.

        Args:
            n_layers: Number of transformer layers
            std: Standard deviation for general weight init
            residual_pattern: Pattern to identify residual connections
            position_embeddings: Optional position embedding module
            position_init_std: Standard deviation for position embedding init
        """
        # Phase 1: Apply standard initialization to all modules
        self._apply_standard_init(std)

        # Phase 2: Re-initialize residual projections with scaling
        residual_std = std / math.sqrt(2 * n_layers)
        self._apply_residual_scaling(residual_pattern, residual_std)

        # Position embeddings use different initialization if they exist
        if position_embeddings is not None:
            if hasattr(position_embeddings, "wpe"):
                # GPT-2 style
                nn.init.normal_(position_embeddings.wpe.weight, std=position_init_std)
            elif hasattr(position_embeddings, "weight"):
                # Standard embedding
                nn.init.normal_(position_embeddings.weight, std=position_init_std)

    def _apply_standard_init(self, std: float) -> None:
        """
        Apply standard initialization to all modules.

        Args:
            std: Standard deviation for weight initialization
        """

        def init_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.apply(init_module)

    def _apply_residual_scaling(
        self,
        residual_pattern: str,
        residual_std: float,
    ) -> None:
        """
        Apply special scaling to residual projection layers.

        Args:
            residual_pattern: Pattern to identify residual projections
            residual_std: Scaled standard deviation for residual projections
        """
        for param_name, param in self.named_parameters():
            if param_name.endswith(residual_pattern):
                torch.nn.init.normal_(param, mean=0.0, std=residual_std)
