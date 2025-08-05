# Standard Library
import time
import typing

# Third Party
import torch


def calculate_throughput_metrics(
    token_counts: typing.Dict[str, int],
    start_time: float,
    num_processes: int,
) -> typing.Dict[str, float]:
    """
    Calculate tokens per second throughput metrics.

    Args:
        token_counts: Token count dictionary
        start_time: Training start time
        num_processes: Number of distributed processes

    Returns:
        Dictionary of throughput metrics
    """
    elapsed_time = time.time() - start_time

    return {
        "per_device_tps": token_counts["total_tokens"] / num_processes / elapsed_time,
        "per_device_tps_including_padding": (
            token_counts["total_tokens_including_padding"] / num_processes / elapsed_time
        ),
    }


def get_memory_metrics() -> typing.Dict[str, float]:
    """
    Get GPU memory usage metrics.

    Returns:
        Dictionary of memory metrics in GiB
    """
    if not torch.cuda.is_available():
        return {}

    return {
        "reserved_mem_GiB": torch.cuda.max_memory_reserved(device=torch.cuda.current_device())
        / 2**30,
        "allocated_mem_GiB": torch.cuda.max_memory_allocated(device=torch.cuda.current_device())
        / 2**30,
    }


def get_learning_rate(lr_scheduler: typing.Any) -> float:
    """
    Get current learning rate from scheduler.

    Args:
        lr_scheduler: Learning rate scheduler

    Returns:
        Current learning rate
    """
    return lr_scheduler.get_last_lr()[0]


def format_metrics_for_logging(
    metrics: typing.Dict[str, float],
    completed_steps: int,
    reduce_loss: str,
) -> str:
    """
    Format metrics for console logging.

    Args:
        metrics: Metrics dictionary
        completed_steps: Number of completed steps
        reduce_loss: Loss reduction strategy

    Returns:
        Formatted string for logging
    """
    lr = metrics["learning_rate"]
    tps = metrics.get("per_device_tps", 0)

    if reduce_loss == "mean":
        loss = metrics["train_loss"]
        return f"  Step: {completed_steps}, LR: {lr}, Loss: {loss}, TPS: {tps}"
    else:
        loss = metrics["train_loss_per_pred_tok"]
        return f"  Step: {completed_steps}, LR: {lr}, Loss/pred_tok: {loss}, TPS: {tps}"
