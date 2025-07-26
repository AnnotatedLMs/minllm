# Standard Library
import time
import typing
from collections import defaultdict

# Third Party
import torch
import torch.nn.functional as F


class MetricsTracker:
    """Tracks and aggregates training metrics.

    Model Effects:
    - Provides visibility into training progress
    - Enables early stopping decisions
    - Helps diagnose training issues

    Core Operations:
    - Accumulates metrics across batches
    - Computes running averages
    - Tracks best values for model selection

    Key Pretraining Metrics:
    - Loss (cross-entropy): Average negative log probability of correct tokens
      - Lower is better: Model assigns higher probability to correct tokens
      - High loss: Model is confused, predictions far from targets
      - Decreasing: Model is learning patterns in data

    - Perplexity: exp(loss), geometric mean of token probabilities
      - Lower is better: Model is more confident in predictions
      - Perplexity of 100 = model thinks there are ~100 equally likely next tokens
      - Perplexity of 10 = model has narrowed down to ~10 likely candidates

    - Gradient norm: L2 norm of all gradients
      - Indicates training stability
      - Spikes suggest exploding gradients
      - Near-zero suggests vanishing gradients

    - Learning rate: Current learning rate from scheduler
      - Tracks warmup and decay phases
      - Helps correlate training dynamics with LR changes

    - Tokens/second: Training throughput
      - Higher is better: Faster training
      - Depends on: batch size, sequence length, model size, hardware
    """

    def __init__(self):
        self.metrics = defaultdict(lambda: {"sum": 0.0, "count": 0})
        self.best_metrics = {}
        self.timer_start = None

    def update(self, **kwargs: float) -> None:
        """Update metrics with new values.

        Args:
            **kwargs: Metric name to value mapping
        """
        for key, value in kwargs.items():
            self.metrics[key]["sum"] += value
            self.metrics[key]["count"] += 1

    def get_average(self, metric: str) -> float:
        """Get average value for a metric.

        Args:
            metric: Name of the metric

        Returns:
            Average value or 0 if metric not tracked
        """
        if metric not in self.metrics or self.metrics[metric]["count"] == 0:
            return 0.0
        return self.metrics[metric]["sum"] / self.metrics[metric]["count"]

    def reset(self) -> None:
        """Reset all metrics for new epoch/evaluation."""
        self.metrics.clear()

    def start_timer(self) -> None:
        """Start timing for throughput metrics."""
        self.timer_start = time.time()

    def end_timer(self, num_tokens: int) -> float:
        """End timing and compute throughput.

        Args:
            num_tokens: Number of tokens processed

        Returns:
            Tokens per second
        """
        if self.timer_start is None:
            return 0.0

        elapsed = time.time() - self.timer_start
        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
        self.timer_start = None
        return tokens_per_sec

    def update_best(self, metric: str, value: float, mode: str = "min") -> bool:
        """Update best value for a metric (used for model selection).

        Args:
            metric: Name of the metric (e.g., "val_loss")
            value: Current value
            mode: "min" or "max" - whether lower or higher is better

        Returns:
            True if this is a new best value
        """
        if metric not in self.best_metrics:
            self.best_metrics[metric] = value
            return True

        is_best = False
        if mode == "min" and value < self.best_metrics[metric]:
            self.best_metrics[metric] = value
            is_best = True
        elif mode == "max" and value > self.best_metrics[metric]:
            self.best_metrics[metric] = value
            is_best = True

        return is_best

    def format_metrics(self, include_best: bool = False) -> str:
        """Format metrics for logging display.

        Args:
            include_best: Whether to show best values

        Returns:
            Formatted string of metrics
        """
        parts = []

        # Define display order for common metrics
        priority_metrics = ["loss", "perplexity", "grad_norm", "lr", "tokens_per_sec"]

        # Show priority metrics first
        for key in priority_metrics:
            if key in self.metrics:
                avg = self.get_average(key)
                if key == "lr":
                    parts.append(f"{key}: {avg:.2e}")
                else:
                    parts.append(f"{key}: {avg:.4f}")

                if include_best and key in self.best_metrics:
                    parts[-1] += f" (best: {self.best_metrics[key]:.4f})"

        # Show any other metrics
        for key in sorted(self.metrics.keys()):
            if key not in priority_metrics:
                avg = self.get_average(key)
                parts.append(f"{key}: {avg:.4f}")

                if include_best and key in self.best_metrics:
                    parts[-1] += f" (best: {self.best_metrics[key]:.4f})"

        return " | ".join(parts)


def compute_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Compute cross-entropy loss using PyTorch's optimized implementation.

    Uses F.cross_entropy which combines log_softmax and nll_loss for efficiency.
    This is especially important for large vocabularies.

    Args:
        logits: Model predictions of shape (batch, seq, vocab)
        targets: Target token ids of shape (batch, seq)
        ignore_index: Token id to ignore in loss computation (e.g. padding)

    Returns:
        Scalar loss tensor
    """
    # Reshape for loss computation
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # Use PyTorch's optimized cross_entropy
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index, reduction="mean")

    return loss


def compute_perplexity(loss: typing.Union[float, torch.Tensor]) -> float:
    """Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss). We clamp to avoid overflow.

    Args:
        loss: Cross-entropy loss value (scalar tensor or float)

    Returns:
        Perplexity value
    """
    if isinstance(loss, torch.Tensor):
        # Clamp to avoid overflow, max perplexity ~1e9
        return torch.exp(loss.clamp(max=20)).item()
    else:
        # For float input
        return min(float("inf"), torch.exp(torch.tensor(loss)).item())


def compute_gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """Compute total gradient norm across all parameters.

    Uses torch.nn.utils.clip_grad_norm_ without clipping to just compute norm.
    This is the standard way to monitor gradient health during training.

    Args:
        model: Model with gradients to compute norm for
        norm_type: Type of norm (default: 2.0 for L2 norm)

    Returns:
        Total gradient norm as float
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float("inf"),  # Don't actually clip
        norm_type=norm_type,
    )
    return total_norm.item()
