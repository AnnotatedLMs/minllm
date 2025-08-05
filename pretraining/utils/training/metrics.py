# Standard Library
import math
import typing

# Third Party
import torch

# Project
from pretraining.utils import torch_utils


def compute_perplexity(loss: typing.Union[float, torch.Tensor]) -> float:
    """Compute perplexity from cross-entropy loss.

    What Perplexity Means for ML Researchers:
    - Perplexity = exp(cross_entropy_loss)
    - Intuition: "How surprised is the model by the test data?"
    - Lower = better (model is less surprised/confused)
    - Perplexity of 100 means model is as confused as if choosing
      randomly from 100 equally likely options
    - Random guessing would give perplexity ≈ vocabulary size

    Why We Clamp:
    - Loss > 20 gives perplexity > 485 million (numerically unstable)
    - Such high values indicate training failure anyway

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
        return min(float("inf"), math.exp(min(loss, 20)))


def compute_gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """Compute total gradient norm across all parameters.

    Why Monitor Gradient Norm:
    - Indicates training stability (spikes = instability)
    - Helps diagnose gradient explosion/vanishing
    - Guides learning rate and gradient clipping decisions

    What the Values Mean:
    - Gradient explosion: Sudden spikes to very large values
    - Gradient vanishing: Drops to near 0
    - After clipping: Should be ≤ clip value
    - Watch for sudden changes rather than absolute values

    L2 Norm Intuition:
    - Square root of sum of squared gradients
    - Measures "total magnitude" of parameter updates
    - Most common choice (norm_type=2.0)

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


def collect_train_step_metrics(
    loss: torch.Tensor,
    ce_loss: typing.Optional[torch.Tensor] = None,
    z_loss: typing.Optional[torch.Tensor] = None,
    moe_aux_loss: typing.Optional[torch.Tensor] = None,
    mtp_losses: typing.Optional[typing.List[torch.Tensor]] = None,
) -> typing.Dict[str, float]:
    """Collect metrics from a training step.

    Args:
        loss: Total loss (already reduced)
        ce_loss: Cross-entropy loss component
        z_loss: Z-loss component (if using)
        moe_aux_loss: MoE auxiliary loss (if using MoE)
        mtp_losses: Multi-token prediction losses (if using MTP)

    Returns:
        Dictionary of metrics with 'train/' prefix
    """
    metrics = {
        "train/total_loss": loss.item(),
        "train/perplexity": compute_perplexity(ce_loss if ce_loss is not None else loss),
    }

    # Add component losses if present
    if ce_loss is not None:
        metrics["train/ce_loss"] = ce_loss.item()

    if z_loss is not None:
        metrics["train/z_loss"] = z_loss.item()

    if moe_aux_loss is not None:
        metrics["train/moe_aux_loss"] = moe_aux_loss.item()

    if mtp_losses is not None:
        for i, mtp_loss in enumerate(mtp_losses):
            metrics[f"train/mtp_loss_{i}"] = mtp_loss.item()

    return metrics


def collect_optimizer_metrics(
    optimizer: torch.optim.Optimizer,
    grad_norm: float,
) -> typing.Dict[str, float]:
    """Collect optimizer-related metrics.

    Args:
        optimizer: The optimizer
        grad_norm: Computed gradient norm

    Returns:
        Dictionary of optimizer metrics
    """
    metrics = {
        "optim/grad_norm": grad_norm,
    }

    # Add learning rates for each param group
    for i, group in enumerate(optimizer.param_groups):
        if i == 0:
            metrics["optim/lr"] = group["lr"]
        else:
            metrics[f"optim/lr_group_{i}"] = group["lr"]

    return metrics


def collect_throughput_metrics(
    tokens_processed: int,
    elapsed_time: float,
    global_step: int,
) -> typing.Dict[str, float]:
    """Collect throughput/speed metrics.

    Understanding Training Throughput:
    - Tokens/sec: Primary efficiency metric for LLM training
    - Higher = more efficient use of compute resources
    - Depends on: model size, batch size, sequence length, hardware

    Optimization Tips:
    - Increase batch size until GPU memory is ~90% utilized
    - Use gradient accumulation if batch won't fit
    - Enable mixed precision (fp16/bf16) for speedup
    - Use Flash Attention for long sequences
    - Profile to find bottlenecks (data loading vs compute)

    Args:
        tokens_processed: Number of tokens in the batch
        elapsed_time: Time taken for the step in seconds
        global_step: Current training step

    Returns:
        Dictionary of throughput metrics
    """
    tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0

    return {
        "throughput/tokens_per_sec": tokens_per_sec,
        "throughput/tokens_per_sec_per_gpu": tokens_per_sec,  # Assuming single GPU for now
        "throughput/steps_per_sec": 1.0 / elapsed_time if elapsed_time > 0 else 0,
        "progress/global_step": global_step,
    }


def collect_system_metrics(
    include_memory: bool = True,
) -> typing.Dict[str, float]:
    """Collect system metrics like GPU memory usage.

    Args:
        include_memory: Whether to include GPU memory metrics

    Returns:
        Dictionary of system metrics
    """
    metrics = {}

    if include_memory:
        peak_memory_mb = torch_utils.peak_gpu_memory(reset=False)
        if peak_memory_mb is not None:
            metrics["system/peak_gpu_memory_mb"] = peak_memory_mb

    return metrics


def collect_eval_metrics(
    loss: float,
    prefix: str = "val",
) -> typing.Dict[str, float]:
    """Collect evaluation metrics.

    Args:
        loss: Average evaluation loss
        prefix: Metric prefix (e.g., 'val', 'test')

    Returns:
        Dictionary of evaluation metrics
    """
    return {
        f"{prefix}/loss": loss,
        f"{prefix}/perplexity": compute_perplexity(loss),
    }


def update_best_metric(
    current_value: float,
    best_value: float,
    mode: str = "min",
) -> typing.Tuple[float, bool]:
    """Update best metric value and check if improved.

    Args:
        current_value: Current metric value
        best_value: Previous best value
        mode: 'min' if lower is better, 'max' if higher is better

    Returns:
        Tuple of (new_best_value, is_improved)
    """
    is_improved = False

    if mode == "min":
        if current_value < best_value:
            best_value = current_value
            is_improved = True
    elif mode == "max":
        if current_value > best_value:
            best_value = current_value
            is_improved = True
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'min' or 'max'")

    return best_value, is_improved
