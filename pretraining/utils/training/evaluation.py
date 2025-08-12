# Standard Library
import typing

# Third Party
import torch
from torch import nn

# Project
from pretraining.utils.training import loss as loss_utils
from pretraining.utils.training import metrics as metrics_module


class Evaluator:
    """Handles model evaluation during training.

    This class is responsible for computing evaluation metrics on validation data.
    It assumes the model is already in eval mode when evaluate() is called.
    """

    def __init__(
        self,
        loss_handler: loss_utils.LossHandler,
        num_eval_batches: int = 50,
    ):
        """Initialize evaluator.

        Args:
            loss_handler: Loss handler for computing losses
            num_eval_batches: Number of batches to evaluate on
        """
        self.loss_handler = loss_handler
        self.num_eval_batches = num_eval_batches

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: typing.Optional[torch.utils.data.DataLoader],
    ) -> typing.Dict[str, float]:
        """Evaluate model and compute metrics.

        Note: This method assumes the model is already in eval mode.
        The caller is responsible for managing model.train()/eval() state.

        Args:
            model: Model to evaluate (should be in eval mode)
            dataloader: Validation dataloader

        Returns:
            Dictionary of evaluation metrics
        """
        if dataloader is None:
            return {"val/loss": float("inf"), "val/perplexity": float("inf")}

        losses = []

        # Create iterator
        data_iter = iter(dataloader)

        for i in range(self.num_eval_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                # If we run out of data, break early
                if i == 0:
                    raise ValueError("Validation dataloader is empty")
                break

            # Move batch to device
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass - model is in eval mode, so no dropout/MTP/aux losses
            model_output = model.forward(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )

            # Compute loss using loss handler for consistency
            # Note: We only care about cross-entropy loss for evaluation
            loss = self.loss_handler.compute_cross_entropy_loss(
                model_output.logits,
                batch["labels"],
                ignore_index=self.loss_handler.config.ignore_index,
            )
            losses.append(loss.item())

        # Compute average metrics
        avg_loss = sum(losses) / len(losses)
        perplexity = metrics_module.compute_perplexity(avg_loss)

        return {
            "val/loss": avg_loss,
            "val/perplexity": perplexity,
        }
