# Standard Library
import time
import typing

# Third Party
import torch
from accelerate import Accelerator

# Project
from posttraining.instruction_tuning.utils.metrics import loss_metrics
from posttraining.instruction_tuning.utils.metrics import performance_metrics
from posttraining.instruction_tuning.utils.metrics import token_metrics


class SFTMetricsTracker:
    """Track training metrics for supervised fine-tuning."""

    def __init__(
        self,
        accelerator: Accelerator,
        logging_steps: typing.Optional[int] = None,
        per_device_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        reduce_loss: str = "mean",
    ):
        """
        Initialize metrics tracker.

        Args:
            accelerator: Accelerator instance
            logging_steps: How often to log metrics
            per_device_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            reduce_loss: Loss reduction strategy ("mean" or "sum")
        """
        self.accelerator = accelerator
        self.logging_steps = logging_steps
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.reduce_loss = reduce_loss

        # Initialize sub-trackers
        self.token_tracker = token_metrics.TokenMetricsTracker(accelerator.device)
        self.loss_tracker = loss_metrics.LossMetricsTracker(reduce_loss)

        # Timing
        self.start_time = time.time()

    def reset_batch_metrics(self):
        """Reset per-batch tracking metrics."""
        self.token_tracker.reset_batch_metrics()

    def reset_logging_metrics(self):
        """Reset per-logging-period metrics."""
        self.token_tracker.reset_period_metrics()
        self.loss_tracker.reset()

    def update_batch_metrics(self, batch: typing.Dict[str, torch.Tensor]):
        """
        Update metrics for a single batch.

        Args:
            batch: Batch dictionary with labels and attention info
        """
        self.token_tracker.update_from_batch(batch)

    def update_loss(self, loss: torch.Tensor):
        """
        Update loss tracking.

        Args:
            loss: Loss tensor
        """
        self.loss_tracker.update(loss)

    def get_logging_metrics(
        self, completed_steps: int, lr_scheduler: typing.Any
    ) -> typing.Dict[str, float]:
        """
        Get metrics for logging.

        Args:
            completed_steps: Number of completed optimization steps
            lr_scheduler: Learning rate scheduler

        Returns:
            Dictionary of metrics to log
        """
        # Gather token metrics
        token_counts = self.token_tracker.gather_metrics(self.accelerator)

        # Calculate token averages
        token_averages = token_metrics.calculate_token_averages(
            token_counts,
            completed_steps,
            self.accelerator,
            self.per_device_batch_size,
            self.gradient_accumulation_steps,
        )

        # Get loss metrics
        loss_metrics = self.loss_tracker.gather_and_compute(
            self.accelerator,
            token_counts,
            self.logging_steps,
            self.gradient_accumulation_steps,
        )

        # Get performance metrics
        throughput_metrics = performance_metrics.calculate_throughput_metrics(
            token_counts,
            self.start_time,
            self.accelerator.num_processes,
        )

        # Get memory metrics
        memory_metrics = performance_metrics.get_memory_metrics()

        # Get learning rate
        lr_metric = {"learning_rate": performance_metrics.get_learning_rate(lr_scheduler)}

        # Reset period counters
        self.token_tracker.reset_period_counters()

        # Combine all metrics
        metrics = {}
        metrics.update(token_counts)
        metrics.update(token_averages)
        metrics.update(loss_metrics)
        metrics.update(throughput_metrics)
        metrics.update(memory_metrics)
        metrics.update(lr_metric)

        return metrics

    def log_metrics(self, metrics: typing.Dict[str, float], completed_steps: int):
        """
        Log metrics to console.

        Args:
            metrics: Metrics dictionary
            completed_steps: Number of completed steps
        """
        log_message = performance_metrics.format_metrics_for_logging(
            metrics,
            completed_steps,
            self.reduce_loss,
        )
        self.accelerator.print(log_message)
