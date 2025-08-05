# Standard Library
import typing

# Third Party
import torch
from accelerate import Accelerator


class LossMetricsTracker:
    """Track loss-related metrics during training."""

    def __init__(self, reduce_loss: str = "mean"):
        """
        Initialize loss metrics tracker.

        Args:
            reduce_loss: Loss reduction strategy ("mean" or "sum")
        """
        self.reduce_loss = reduce_loss
        self.total_loss = 0

    def update(self, loss: torch.Tensor):
        """
        Update loss tracking.

        Args:
            loss: Loss tensor from current batch
        """
        self.total_loss += loss.detach().float()

    def gather_and_compute(
        self,
        accelerator: Accelerator,
        token_counts: typing.Dict[str, int],
        logging_steps: int,
        gradient_accumulation_steps: int,
    ) -> typing.Dict[str, float]:
        """
        Gather loss across devices and compute metrics.

        Args:
            accelerator: Accelerator for distributed gathering
            token_counts: Token count dictionary
            logging_steps: Number of logging steps
            gradient_accumulation_steps: Gradient accumulation steps

        Returns:
            Dictionary of loss metrics
        """
        # Gather loss across devices
        sum_loss = accelerator.gather(self.total_loss).sum().item()

        # Compute metrics based on reduction type
        if self.reduce_loss == "mean":
            metrics = self._compute_mean_loss_metrics(
                sum_loss,
                logging_steps,
                gradient_accumulation_steps,
                accelerator.num_processes,
            )
        else:
            metrics = self._compute_sum_loss_metrics(
                sum_loss,
                token_counts,
                logging_steps,
                accelerator.num_processes,
            )

        # Reset for next period
        self.reset()

        return metrics

    def reset(self):
        """Reset loss accumulator."""
        self.total_loss = 0

    def _compute_mean_loss_metrics(
        self,
        sum_loss: float,
        logging_steps: int,
        gradient_accumulation_steps: int,
        num_processes: int,
    ) -> typing.Dict[str, float]:
        """
        Compute metrics for mean-reduced loss.

        Args:
            sum_loss: Total loss across all devices
            logging_steps: Number of logging steps
            gradient_accumulation_steps: Gradient accumulation steps
            num_processes: Number of distributed processes

        Returns:
            Dictionary with train_loss
        """
        total_fwd_passes = logging_steps * gradient_accumulation_steps * num_processes
        avg_loss = sum_loss / total_fwd_passes

        return {"train_loss": avg_loss}

    def _compute_sum_loss_metrics(
        self,
        sum_loss: float,
        token_counts: typing.Dict[str, int],
        logging_steps: int,
        num_processes: int,
    ) -> typing.Dict[str, float]:
        """
        Compute metrics for sum-reduced loss.

        Args:
            sum_loss: Total loss across all devices
            token_counts: Token count dictionary
            logging_steps: Number of logging steps
            num_processes: Number of distributed processes

        Returns:
            Dictionary with various loss metrics
        """
        avg_loss = sum_loss / token_counts["total_tokens_this_period"]
        avg_loss_per_pred_tok = sum_loss / token_counts["pred_tokens_this_period"]
        total_optim_steps = logging_steps * num_processes
        avg_sum_loss = sum_loss / total_optim_steps

        return {
            "train_sum_loss": avg_sum_loss,
            "train_loss_per_total_tok": avg_loss,
            "train_loss_per_pred_tok": avg_loss_per_pred_tok,
        }


def get_loss_for_logging(
    metrics: typing.Dict[str, float],
    reduce_loss: str,
) -> typing.Tuple[str, float]:
    """
    Get the appropriate loss metric for console logging.

    Args:
        metrics: Metrics dictionary
        reduce_loss: Loss reduction strategy

    Returns:
        Tuple of (loss_name, loss_value)
    """
    if reduce_loss == "mean":
        return "Loss", metrics["train_loss"]
    else:
        return "Loss/pred_tok", metrics["train_loss_per_pred_tok"]
