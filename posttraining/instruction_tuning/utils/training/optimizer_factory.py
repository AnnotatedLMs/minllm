# Standard Library
import typing

# Third Party
import torch
import torch.nn as nn
from transformers import PreTrainedModel


class OptimizerFactory:
    """Factory for creating optimizers with SFT-specific configurations.

    Provides a clean interface for creating optimizers with proper parameter
    grouping and weight decay handling, following best practices for fine-tuning.
    """

    @staticmethod
    def create_optimizer_groups(
        model: typing.Union[nn.Module, PreTrainedModel],
        weight_decay: float = 0.0,
        no_decay_keywords: typing.Optional[typing.List[str]] = None,
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """Create parameter groups with decay/no-decay separation.

        Args:
            model: Model to get parameters from
            weight_decay: Weight decay value for parameters
            no_decay_keywords: Keywords to identify no-decay parameters

        Returns:
            List of parameter groups for optimizer
        """
        if no_decay_keywords is None:
            no_decay_keywords = ["bias", "layer_norm.weight", "layernorm.weight"]

        # Separate parameters into decay/no-decay groups
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if any(nd_kw in name for nd_kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    @staticmethod
    def create_adamw(
        model: typing.Union[nn.Module, PreTrainedModel],
        learning_rate: float,
        weight_decay: float = 0.01,
        betas: typing.Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        fused: bool = False,
        use_qlora: bool = False,
    ) -> torch.optim.Optimizer:
        """Create AdamW optimizer with proper configuration.

        Args:
            model: Model to optimize
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            betas: Adam beta parameters
            eps: Adam epsilon
            fused: Use fused implementation if available
            use_qlora: Use 8-bit AdamW for QLoRA

        Returns:
            Configured optimizer
        """
        # Get parameter groups
        param_groups = OptimizerFactory.create_optimizer_groups(model, weight_decay)

        if use_qlora:
            # Use bitsandbytes 8-bit optimizer for QLoRA
            try:
                # Third Party
                from bitsandbytes.optim import AdamW

                return AdamW(
                    param_groups,
                    lr=learning_rate,
                    betas=betas,
                    eps=eps,
                    optim_bits=8,
                    is_paged=True,
                )
            except ImportError:
                raise ImportError(
                    "bitsandbytes required for QLoRA. Install with: pip install bitsandbytes"
                )
        else:
            # Standard PyTorch AdamW
            return torch.optim.AdamW(
                param_groups,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                fused=fused and torch.cuda.is_available(),
            )

    @staticmethod
    def create_sgd(
        model: typing.Union[nn.Module, PreTrainedModel],
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> torch.optim.Optimizer:
        """Create SGD optimizer with momentum.

        Args:
            model: Model to optimize
            learning_rate: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay
            nesterov: Use Nesterov momentum

        Returns:
            Configured optimizer
        """
        param_groups = OptimizerFactory.create_optimizer_groups(model, weight_decay)

        return torch.optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
        )
