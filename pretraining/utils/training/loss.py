# Standard Library
import typing

# Third Party
import torch


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
        model_output: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]],
    ) -> typing.Dict[str, torch.Tensor]:
        """Extract losses from model output.

        Model training_forward can return:
        - Single tensor: Just the main loss
        - Tuple of 2: (main_loss, logits)
        - Tuple of 3: (main_loss, logits, extras) where extras has auxiliary losses

        Args:
            model_output: Output from model.training_forward()

        Returns:
            Dictionary mapping loss names to tensors
        """
        losses = {}

        # Handle single tensor output (just loss)
        if isinstance(model_output, torch.Tensor):
            losses["loss"] = model_output
            return losses

        # Handle tuple output
        if not isinstance(model_output, tuple):
            raise ValueError(f"Expected tensor or tuple, got {type(model_output)}")

        # First element is always main loss
        losses["loss"] = model_output[0]

        # Third element (if present) contains extras
        if len(model_output) >= 3 and model_output[2] is not None:
            extras = model_output[2]

            # Handle auxiliary losses (e.g., from MoE)
            if "aux_losses" in extras:
                aux_losses = extras["aux_losses"]
                if isinstance(aux_losses, list) and aux_losses:
                    # Sum all auxiliary losses
                    losses["aux_loss"] = sum(aux_losses)
                elif isinstance(aux_losses, torch.Tensor):
                    losses["aux_loss"] = aux_losses

            # Handle MTP losses
            if "mtp_losses" in extras:
                mtp_losses = extras["mtp_losses"]
                # MTP returns losses for each future position
                for i, mtp_loss in enumerate(mtp_losses):
                    if mtp_loss is not None:
                        losses[f"mtp_loss_{i + 1}"] = mtp_loss

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
