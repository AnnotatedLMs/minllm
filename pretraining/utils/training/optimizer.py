# Standard Library
import inspect
import logging
import typing

# Third Party
import torch
from torch import nn
from torch import optim

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """Factory for creating optimizers used in LLM training.

    Model Effects:
    - Enables learning by updating model parameters based on gradients
    - Controls convergence speed and training stability
    - Prevents overfitting through weight regularization

    Core Operations:
    - Groups parameters by dimension or name for differential treatment
    - Creates AdamW optimizer with architecture-specific configurations
    - Applies weight decay selectively based on parameter type

    Training Operations (what optimizers do):
    1. Receive gradients computed during backpropagation
    2. Store gradient history for momentum/adaptive learning rates
    3. Scale gradients by the learning rate
    4. Apply momentum - blend current gradient with past gradients
    5. Apply weight decay - reduce parameter values slightly
    6. Update parameters - subtract processed gradients from current values
    7. Clear gradients for next iteration
    """

    @staticmethod
    def create_parameter_groups_by_dimension(
        model: nn.Module,
        weight_decay: float,
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """
        Create parameter groups based on tensor dimension.

        Strategy:
        - 1D parameters (biases, layernorms) get no weight decay
        - 2D+ parameters (weight matrices) get weight decay
        """
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

    @staticmethod
    def create_parameter_groups_by_name(
        model: nn.Module,
        weight_decay: float,
        no_decay_patterns: typing.List[str],
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """
        Create parameter groups based on parameter names.

        Strategy:
        - Parameters matching no_decay_patterns get no weight decay
        - All other parameters get weight decay
        """
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

        decay_params = [
            p for n, p in param_dict.items() if not any(nd in n for nd in no_decay_patterns)
        ]
        nodecay_params = [
            p for n, p in param_dict.items() if any(nd in n for nd in no_decay_patterns)
        ]

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

    @staticmethod
    def create_adamw(
        model: nn.Module,
        learning_rate: float,
        weight_decay: float = 0.1,
        betas: typing.Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        parameter_grouping: typing.Literal["dimension", "name"] = "dimension",
        no_decay_patterns: typing.List[str] = None,
        device_type: str = "cuda",
    ) -> optim.AdamW:
        """
        Create AdamW optimizer with appropriate parameter groups.

        AdamW is a standard optimizer for LLM training:

        Parameter grouping strategies:
        - "dimension": Parameters grouped by tensor dimension
          * Used by: GPT-2 (nanoGPT), Llama
          * Rule: 1D tensors (biases, norms) no decay, 2D+ tensors decay
        - "name": Parameters grouped by name patterns
          * Used by: DeepSeek
          * Rule: Names with patterns in no_decay_patterns get no decay
          * Common patterns: ['bias', 'LayerNorm.weight']

        Common hyperparameters:
        - GPT-2: betas=(0.9, 0.95), weight_decay=0.1
        - Llama 3: weight_decay=0.1*lr (dynamically scaled)
        - DeepSeek-V3: betas=(0.9, 0.95), weight_decay=0.1

        Args:
            model: Model to optimize
            learning_rate: Learning rate
            weight_decay: Weight decay coefficient (default: 0.1)
            betas: Adam beta parameters (default: (0.9, 0.95))
            eps: Adam epsilon (default: 1e-8)
            parameter_grouping: "dimension" or "name" based grouping
            no_decay_patterns: Patterns for name-based grouping (required if grouping="name")
            device_type: Device for fused optimizer check

        Returns:
            Configured AdamW optimizer
        """
        if parameter_grouping == "name" and no_decay_patterns is None:
            raise ValueError(
                "no_decay_patterns must be provided when using name-based grouping. ie. ['bias', 'LayerNorm.weight']"
            )

        if parameter_grouping == "dimension":
            param_groups = OptimizerFactory.create_parameter_groups_by_dimension(
                model, weight_decay
            )
        elif parameter_grouping == "name":
            param_groups = OptimizerFactory.create_parameter_groups_by_name(
                model, weight_decay, no_decay_patterns
            )
        else:
            raise ValueError(f"Unknown parameter grouping: {parameter_grouping}")

        # Check for fused optimizer (PyTorch 2.0+)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        optimizer = optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            fused=use_fused,
        )

        logger.info(
            f"Created AdamW optimizer (fused={use_fused}, betas={betas}, "
            f"weight_decay={weight_decay}, grouping={parameter_grouping})"
        )
        return optimizer
