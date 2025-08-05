# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn


def shift_for_next_token_prediction(
    logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
    labels: jaxtyping.Int[torch.Tensor, "batch seq"],
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Shift logits and labels for next-token prediction.

    In autoregressive language modeling, we predict token n+1 from tokens 1...n,
    so we need to shift the sequences.

    Args:
        logits: Model predictions [batch, seq, vocab]
        labels: Target labels [batch, seq]

    Returns:
        Tuple of (shifted_logits, shifted_labels)
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return shift_logits, shift_labels


def apply_mean_reduction(
    logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
    labels: jaxtyping.Int[torch.Tensor, "batch seq"],
    vocab_size: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Apply mean reduction strategy for cross-entropy loss.

    This computes the average loss per valid token.

    Args:
        logits: Model predictions
        labels: Target labels
        vocab_size: Size of vocabulary
        ignore_index: Token ID to ignore

    Returns:
        Mean loss value
    """
    # Shift for next-token prediction
    shift_logits, shift_labels = shift_for_next_token_prediction(logits, labels)

    # Flatten for loss computation
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    # Compute cross-entropy loss with mean reduction
    loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

    return loss_fct(shift_logits, shift_labels)


def apply_sum_reduction(
    logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
    labels: jaxtyping.Int[torch.Tensor, "batch seq"],
    vocab_size: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Apply sum reduction strategy for cross-entropy loss.

    Sum reduction ensures equal weighting of all tokens across the entire
    dataset when using gradient accumulation, rather than equal weighting
    of batches.

    Args:
        logits: Model predictions
        labels: Target labels
        vocab_size: Size of vocabulary
        ignore_index: Token ID to ignore

    Returns:
        Sum of losses over all valid tokens
    """
    # Clone labels to avoid modifying original
    labels = labels.clone()

    # Shift for next-token prediction
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]

    # Create mask for valid tokens
    loss_mask = labels != ignore_index

    # Replace ignore_index with 0 to avoid indexing errors
    labels[labels == ignore_index] = 0

    # Compute per-token loss
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    # Flatten tensors
    shift_logits = logits.reshape(-1, vocab_size)
    shift_labels = labels.reshape(-1)

    # Compute loss
    per_token_loss = loss_fct(shift_logits, shift_labels)

    # Apply mask and sum
    per_token_loss = per_token_loss.reshape_as(labels)
    masked_loss = per_token_loss * loss_mask

    return masked_loss.sum()


def create_loss_mask(
    labels: jaxtyping.Int[torch.Tensor, "batch seq"],
    ignore_index: int = -100,
) -> jaxtyping.Bool[torch.Tensor, "batch seq"]:
    """
    Create a mask for valid tokens in loss computation.

    Args:
        labels: Target labels
        ignore_index: Token ID to ignore

    Returns:
        Boolean mask where True indicates valid tokens
    """
    return labels != ignore_index


def count_valid_tokens(
    labels: jaxtyping.Int[torch.Tensor, "batch seq"],
    ignore_index: int = -100,
) -> int:
    """
    Count the number of valid tokens for loss computation.

    Args:
        labels: Target labels
        ignore_index: Token ID to ignore

    Returns:
        Number of valid tokens
    """
    loss_mask = create_loss_mask(labels, ignore_index)
    return loss_mask.sum().item()
