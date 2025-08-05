# Standard Library
import typing

# Third Party
import torch
from accelerate import Accelerator


class TokenMetricsTracker:
    """Track token-related metrics during training."""

    def __init__(self, device: torch.device):
        """
        Initialize token metrics tracker.

        Args:
            device: Device to place tensors on
        """
        self.device = device
        self.reset_batch_metrics()
        self.reset_period_metrics()

    def reset_batch_metrics(self):
        """Reset per-batch token counters."""
        self.total_tokens = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.pred_tokens = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.tokens_including_padding = torch.tensor(0, dtype=torch.int64, device=self.device)

    def reset_period_metrics(self):
        """Reset per-logging-period token counters."""
        self.total_tokens_this_period = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.pred_tokens_this_period = torch.tensor(0, dtype=torch.int64, device=self.device)

    def update_from_batch(self, batch: typing.Dict[str, torch.Tensor]):
        """
        Update token counts from a batch.

        Args:
            batch: Batch dictionary with labels and attention info
        """
        # Count prediction tokens (non-masked)
        pred_tokens_in_batch = (batch["labels"] != -100).sum()
        self.pred_tokens += pred_tokens_in_batch
        self.pred_tokens_this_period += pred_tokens_in_batch

        # Count total tokens based on what's available
        tokens_in_batch = count_tokens_in_batch(batch)
        self.total_tokens += tokens_in_batch
        self.total_tokens_this_period += tokens_in_batch

        # Count tokens including padding
        if "attention_mask" in batch:
            self.tokens_including_padding += batch["attention_mask"].numel()
        else:
            self.tokens_including_padding += tokens_in_batch

    def gather_metrics(self, accelerator: Accelerator) -> typing.Dict[str, int]:
        """
        Gather token metrics across all devices.

        Args:
            accelerator: Accelerator for distributed gathering

        Returns:
            Dictionary of gathered token counts
        """
        return {
            "total_tokens": accelerator.gather(self.total_tokens).sum().item(),
            "total_pred_tokens": accelerator.gather(self.pred_tokens).sum().item(),
            "total_tokens_including_padding": accelerator.gather(self.tokens_including_padding)
            .sum()
            .item(),
            "total_tokens_this_period": accelerator.gather(self.total_tokens_this_period)
            .sum()
            .item(),
            "pred_tokens_this_period": accelerator.gather(self.pred_tokens_this_period)
            .sum()
            .item(),
        }

    def reset_period_counters(self):
        """Reset period-specific counters after logging."""
        self.total_tokens_this_period.zero_()
        self.pred_tokens_this_period.zero_()


def count_tokens_in_batch(batch: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Count actual tokens in a batch.

    Args:
        batch: Batch dictionary

    Returns:
        Number of tokens

    Raises:
        ValueError: If no supported token counting method found
    """
    if "attention_mask" in batch:
        return batch["attention_mask"].sum()
    elif "position_ids" in batch:
        return batch["position_ids"].numel()
    elif "cu_seq_lens_q" in batch:
        return batch["cu_seq_lens_q"][-1]
    else:
        raise ValueError("Expected attention_mask, position_ids, or cu_seq_lens_q in batch")


def calculate_token_averages(
    token_counts: typing.Dict[str, int],
    completed_steps: int,
    accelerator: Accelerator,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> typing.Dict[str, float]:
    """
    Calculate average token metrics per batch.

    Args:
        token_counts: Token count dictionary
        completed_steps: Number of completed optimization steps
        accelerator: Accelerator instance
        per_device_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps

    Returns:
        Dictionary of average metrics
    """
    divisor = (
        accelerator.num_processes
        * per_device_batch_size
        * gradient_accumulation_steps
        * completed_steps
    )

    return {
        "avg_tokens_per_batch": token_counts["total_tokens"] / divisor,
        "avg_tokens_per_batch_including_padding": (
            token_counts["total_tokens_including_padding"] / divisor
        ),
        "avg_pred_tokens_per_batch": token_counts["total_pred_tokens"] / divisor,
    }
