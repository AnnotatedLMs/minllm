# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

# Project
from posttraining.instruction_tuning.common.patterns.loss import masked_ce_loss
from posttraining.instruction_tuning.utils import metrics


def prepare_dataloader_for_epoch(
    dataloader: DataLoader,
    epoch: int,
    resume_batch_idx: int,
    accelerator: Accelerator,
) -> DataLoader:
    """
    Prepare dataloader for training epoch.

    Args:
        dataloader: Training dataloader
        epoch: Current epoch number
        resume_batch_idx: Batch index to resume from
        accelerator: Accelerator instance

    Returns:
        Prepared dataloader
    """
    # Set epoch for proper data shuffling
    if hasattr(dataloader, "set_epoch"):
        dataloader.set_epoch(epoch)

    # Skip batches if resuming
    if resume_batch_idx > 0:
        return accelerator.skip_first_batches(dataloader, resume_batch_idx)

    return dataloader


def compute_training_loss(
    model_outputs: typing.Any,
    batch: typing.Dict[str, jaxtyping.Int[torch.Tensor, "batch seq"]],
    vocab_size: int,
    reduce_loss: str,
) -> jaxtyping.Float[torch.Tensor, ""]:
    """
    Compute training loss based on reduction strategy.

    Args:
        model_outputs: Model forward pass outputs
        batch: Input batch
        vocab_size: Vocabulary size
        reduce_loss: Loss reduction strategy ("mean" or "sum")

    Returns:
        Computed loss
    """
    if reduce_loss == "mean":
        return model_outputs.loss
    else:
        # Manual loss computation for sum reduction
        return masked_ce_loss.compute_sft_loss(
            model_outputs.logits, batch["labels"], vocab_size, reduction="sum"
        )


def perform_forward_pass(
    model: PreTrainedModel,
    batch: typing.Dict[str, torch.Tensor],
    use_cache: bool = False,
) -> typing.Any:
    """
    Perform forward pass through the model.

    Args:
        model: Model to run forward pass
        batch: Input batch
        use_cache: Whether to use KV cache

    Returns:
        Model outputs
    """
    return model(**batch, use_cache=use_cache)


def perform_backward_pass(
    accelerator: Accelerator,
    loss: jaxtyping.Float[torch.Tensor, ""],
    model: PreTrainedModel,
    clip_grad_norm: float,
) -> None:
    """
    Perform backward pass and gradient clipping.

    Args:
        accelerator: Accelerator instance
        loss: Computed loss
        model: Model being trained
        clip_grad_norm: Gradient clipping value
    """
    # Backward pass
    accelerator.backward(loss)

    # Gradient clipping
    if accelerator.sync_gradients and clip_grad_norm > 0:
        accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)


def perform_optimizer_step(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: typing.Any,
) -> None:
    """
    Perform optimizer step and learning rate update.

    Args:
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
    """
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()


def should_log_metrics(
    completed_steps: int,
    logging_steps: typing.Optional[int],
) -> bool:
    """
    Check if metrics should be logged at this step.

    Args:
        completed_steps: Number of completed steps
        logging_steps: Logging frequency

    Returns:
        Whether to log metrics
    """
    return logging_steps and completed_steps % logging_steps == 0


def should_save_checkpoint(
    completed_steps: int,
    checkpointing_steps: typing.Union[int, str],
) -> bool:
    """
    Check if checkpoint should be saved at this step.

    Args:
        completed_steps: Number of completed steps
        checkpointing_steps: Checkpointing frequency

    Returns:
        Whether to save checkpoint
    """
    return isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0


def log_training_metrics(
    metrics_tracker: metrics.SFTMetricsTracker,
    completed_steps: int,
    lr_scheduler: typing.Any,
    accelerator: Accelerator,
    report_to: typing.Optional[str],
) -> None:
    """
    Log training metrics.

    Args:
        metrics_tracker: Metrics tracker instance
        completed_steps: Number of completed steps
        lr_scheduler: Learning rate scheduler
        accelerator: Accelerator instance
        report_to: Where to report metrics
    """
    metrics = metrics_tracker.get_logging_metrics(completed_steps, lr_scheduler)
    metrics_tracker.log_metrics(metrics, completed_steps)

    if accelerator.is_main_process and report_to:
        accelerator.log(metrics, step=completed_steps)
