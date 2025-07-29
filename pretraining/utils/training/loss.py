# Standard Library
import typing

# Third Party
import torch

# Project
from pretraining.common.base import outputs


class LossHandler:
    """Handles loss extraction and aggregation from model outputs.

    Model Effects:
    - Provides unified interface for different model architectures
    - Enables auxiliary loss support (MoE, MTP)
    - Simplifies trainer implementation

    Core Operations:
    - Extracts losses from model training_forward outputs
    - Aggregates multiple loss components
    - Tracks individual loss components for logging

    Loss Types in LLM Pretraining:
    - Main loss: Standard cross-entropy for next token prediction
    - Auxiliary loss (MoE): Load balancing loss to ensure expert utilization
    - MTP losses: Multi-token prediction losses for future tokens
    - Regularization: Sometimes L2 or other penalties (handled by optimizer)
    """

    @staticmethod
    def extract_losses(
        model_output: outputs.TrainingOutput,
    ) -> typing.Dict[str, torch.Tensor]:
        """Extract losses from model output.

        Takes the structured TrainingOutput from model.training_forward() and extracts
        all loss components into a flat dictionary for aggregation and logging.

        Args:
            model_output: TrainingOutput containing:
                - loss: Main language modeling loss (cross-entropy) [scalar tensor]
                - mtp_losses: Optional list of multi-token prediction losses
                              [list of scalar tensors, one per future position]
                - aux_losses: Optional list of auxiliary losses (e.g., MoE load balancing)
                              [list of scalar tensors, one per layer with MoE]

        Returns:
            Dictionary mapping loss names to tensors:
                - "loss": Main LM loss
                - "mtp_loss_1", "mtp_loss_2", etc.: Individual MTP losses by position
                - "aux_loss": Sum of all auxiliary losses (for MoE load balancing)
        """
        losses = {}
        losses["loss"] = model_output.loss

        # Add MTP losses if present
        if model_output.mtp_losses:
            for i, mtp_loss in enumerate(model_output.mtp_losses):
                losses[f"mtp_loss_{i + 1}"] = mtp_loss

        # Add auxiliary losses if present
        if model_output.aux_losses:
            # Sum all auxiliary losses
            losses["aux_loss"] = sum(model_output.aux_losses)

        return losses

    @staticmethod
    def aggregate_losses(
        losses: typing.Dict[str, torch.Tensor],
        weights: typing.Optional[typing.Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Aggregate multiple losses into single training loss.

        Args:
            losses: Dictionary of loss components
            weights: Optional weights for each loss component

        Returns:
            Aggregated loss for backpropagation
        """
        if weights is None:
            weights = {}

        # Default weights
        default_weights = {
            "loss": 1.0,
            "aux_loss": 0.01,  # MoE auxiliary loss weight
            "mtp_loss_1": 0.65,  # MTP weights from DeepSeek paper
            "mtp_loss_2": 0.2,
            "mtp_loss_3": 0.15,
        }

        total_loss = torch.tensor(0.0, device=losses["loss"].device)

        for name, loss_tensor in losses.items():
            weight = weights.get(name, default_weights.get(name, 1.0))
            total_loss = total_loss + weight * loss_tensor

        return total_loss

    @staticmethod
    def format_losses_for_logging(
        losses: typing.Dict[str, torch.Tensor],
    ) -> typing.Dict[str, float]:
        """Convert loss tensors to float values for logging.

        Args:
            losses: Dictionary of loss tensors

        Returns:
            Dictionary of loss values as floats
        """
        return {name: loss.item() for name, loss in losses.items()}
