# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models import inputs
from pretraining.common.models import outputs
from pretraining.configs.training import loss_configs


class LossHandler:
    """Handles loss computation and aggregation from model outputs.

    Model Effects:
    - Provides unified interface for different model architectures
    - Enables auxiliary loss support (MoE, MTP)
    - Simplifies trainer implementation

    Core Operations:
    - Computes losses from model logits
    - Aggregates multiple loss components with configurable weights
    - Tracks individual loss components for logging

    Loss Types in LLM Pretraining:
    - Cross-entropy loss: Standard next token prediction loss
    - MTP losses: Multi-token prediction losses for future tokens
    - Auxiliary loss (MoE): Load balancing loss from MoE layers
    - Z-loss: Optional logit regularization for training stability
    """

    def __init__(self, loss_config: loss_configs.LossConfig):
        """Initialize with loss configuration.

        Args:
            loss_config: Configuration with loss weights and options
        """
        self.config = loss_config

    def compute_cross_entropy_loss(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
        labels: jaxtyping.Int[torch.Tensor, "batch seq"],
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for language modeling.

        Args:
            logits: Model output logits [batch, seq, vocab]
            labels: Target token indices [batch, seq]
            ignore_index: Index to ignore in loss computation

        Returns:
            Cross-entropy loss (scalar tensor)
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute loss
        if self.config.use_fused_loss:
            # TODO: Use fused cross-entropy from flash-attn if available
            loss = nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=ignore_index
            )
        else:
            loss = nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=ignore_index
            )

        return loss

    def compute_z_loss(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
    ) -> torch.Tensor:
        """Compute z-loss for training stability (from OLMo).

        Z-loss encourages logits to stay close to zero, improving stability.

        Args:
            logits: Model output logits [batch, seq, vocab]

        Returns:
            Z-loss (scalar tensor)
        """
        # Z-loss is the squared norm of logits
        z_loss = torch.square(logits).mean()
        return z_loss

    def compute_mtp_losses(
        self,
        mtp_logits: typing.List[jaxtyping.Float[torch.Tensor, "batch seq vocab"]],
        mtp_targets: jaxtyping.Int[torch.Tensor, "batch depth seq"],
        ignore_index: int = -100,
    ) -> typing.List[torch.Tensor]:
        """Compute multi-token prediction losses.

        Args:
            mtp_logits: List of logits for each future position
            mtp_targets: Target tokens for each depth [batch, depth, seq]
            ignore_index: Index to ignore in loss

        Returns:
            List of MTP losses for each depth
        """
        mtp_losses = []

        for d, depth_logits in enumerate(mtp_logits):
            if d < mtp_targets.shape[1]:
                # Get targets for this depth
                depth_targets = mtp_targets[:, d, :]

                # Flatten for loss computation
                depth_logits_flat = depth_logits.reshape(-1, depth_logits.size(-1))
                depth_targets_flat = depth_targets.reshape(-1)

                # Compute loss for this depth
                depth_loss = nn.functional.cross_entropy(
                    depth_logits_flat, depth_targets_flat, ignore_index=ignore_index
                )
                mtp_losses.append(depth_loss)

        return mtp_losses

    def compute_losses(
        self,
        model_output: outputs.ForwardOutput,
        training_inputs: inputs.TrainingInputs,
    ) -> typing.Dict[str, torch.Tensor]:
        """Compute all losses from model output and inputs.

        Args:
            model_output: ForwardOutput containing logits, mtp_logits, and aux losses
            training_inputs: TrainingInputs containing labels and targets

        Returns:
            Dictionary mapping loss names to tensors
        """
        losses = {}

        # Compute cross-entropy loss from main logits
        losses["cross_entropy"] = self.compute_cross_entropy_loss(
            model_output.logits,
            training_inputs.labels,
            ignore_index=self.config.ignore_index,
        )

        # Compute z-loss if enabled
        if self.config.z_loss_weight is not None and self.config.z_loss_weight > 0:
            losses["z_loss"] = self.compute_z_loss(model_output.logits)

        # Compute MTP losses if present
        if model_output.mtp_logits is not None and training_inputs.mtp_targets is not None:
            mtp_losses = self.compute_mtp_losses(
                model_output.mtp_logits,
                training_inputs.mtp_targets,
                ignore_index=self.config.ignore_index,
            )
            # Store individual MTP losses for logging
            for i, mtp_loss in enumerate(mtp_losses):
                losses[f"mtp_loss_{i + 1}"] = mtp_loss

        # Add auxiliary losses if present
        if model_output.aux_losses:
            # Sum all auxiliary losses from MoE
            losses["aux_loss"] = sum(model_output.aux_losses)

        return losses

    def aggregate_losses(
        self,
        losses: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate multiple losses into single training loss.

        Args:
            losses: Dictionary of loss components

        Returns:
            Aggregated loss for backpropagation
        """
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)

        # Apply weights from config
        if "cross_entropy" in losses:
            total_loss = total_loss + self.config.cross_entropy_weight * losses["cross_entropy"]

        if "z_loss" in losses and self.config.z_loss_weight is not None:
            total_loss = total_loss + self.config.z_loss_weight * losses["z_loss"]

        if "aux_loss" in losses and self.config.moe_aux_loss_weight is not None:
            total_loss = total_loss + self.config.moe_aux_loss_weight * losses["aux_loss"]

        # Apply MTP weight to averaged MTP losses
        mtp_losses = [losses[k] for k in losses if k.startswith("mtp_loss_")]
        if mtp_losses and self.config.mtp_loss_weight is not None:
            # Average MTP losses as per DeepSeek-V3 paper
            avg_mtp_loss = sum(mtp_losses) / len(mtp_losses)
            total_loss = total_loss + self.config.mtp_loss_weight * avg_mtp_loss

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
