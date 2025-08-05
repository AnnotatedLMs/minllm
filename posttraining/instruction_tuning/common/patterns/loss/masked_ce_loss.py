# Standard Library

# Third Party
import jaxtyping
import torch
import torch.nn as nn

# Project
from posttraining.instruction_tuning.common.patterns.loss import reduction


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with selective token masking for instruction tuning.

    Core Concept:
    In SFT, we only want the model to learn to generate responses, not to
    reproduce the instructions. This loss function ignores tokens marked
    with a special ignore_index (typically -100), allowing us to mask out
    instruction tokens while computing loss only on response tokens.

    Why This Matters:
    1. Prevents overfitting to instruction format
    2. Focuses learning capacity on response generation
    3. Maintains instruction understanding without memorization

    Example:
    Input: "Translate to French: Hello world"
    Response: "Bonjour le monde"

    During training:
    - "Translate to French: Hello world" tokens → labels = -100 (masked)
    - "Bonjour le monde" tokens → labels = actual token IDs (loss computed)

    This ensures the model learns to generate "Bonjour le monde" given
    the instruction, rather than learning to reproduce the entire sequence.
    """

    def __init__(self, vocab_size: int, reduction: str = "mean", ignore_index: int = -100):
        """
        Initialize the loss module.

        Args:
            vocab_size: Size of the vocabulary
            reduction: "mean" or "sum" reduction
            ignore_index: Token ID to ignore in loss computation
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
        labels: jaxtyping.Int[torch.Tensor, "batch seq"],
    ) -> torch.Tensor:
        """
        Compute masked cross-entropy loss.

        Args:
            logits: Model predictions
            labels: Target labels (with ignore_index for masked tokens)

        Returns:
            Loss value
        """
        if self.reduction == "mean":
            return reduction.apply_mean_reduction(
                logits, labels, self.vocab_size, self.ignore_index
            )
        else:
            # For "sum" or other reductions, use the original implementation
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for loss computation
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=self.ignore_index)
            loss = loss_fct(shift_logits, shift_labels)

            return loss


class SumCrossEntropyLoss(nn.Module):
    """Sum-reduced cross-entropy loss for consistent gradient accumulation.

    Core Insight:
    When using gradient accumulation with mean reduction, each batch
    contributes equally to the gradient regardless of how many valid tokens
    it contains. This creates inconsistent training dynamics where batches
    with fewer response tokens have disproportionate influence.

    Why Sum Reduction Matters:
    1. Equal Token Weighting: Each token contributes equally to gradients
    2. Consistent Learning: Training dynamics don't depend on batch composition
    3. Better Convergence: More stable optimization, especially with varying sequence lengths

    Mathematical Difference:
    - Mean: loss = sum(token_losses) / num_batches
    - Sum: loss = sum(token_losses) / num_tokens

    Example Impact:
    Batch 1: 100 response tokens, loss = 2.0
    Batch 2: 10 response tokens, loss = 2.0

    With mean reduction: Both batches contribute equally (bad!)
    With sum reduction: Batch 1 contributes 10x more (good!)

    Reference: https://github.com/huggingface/transformers/issues/24725

    This is particularly important in SFT where response lengths vary
    significantly across examples (e.g., "Hi" vs. a long explanation).
    """

    def __init__(self, vocab_size: int, ignore_index: int = -100):
        """
        Initialize the loss module.

        Args:
            vocab_size: Size of the vocabulary
            ignore_index: Token ID to ignore in loss computation
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
        labels: jaxtyping.Int[torch.Tensor, "batch seq"],
    ) -> torch.Tensor:
        """
        Compute sum-reduced cross-entropy loss.

        Args:
            logits: Model predictions
            labels: Target labels (with ignore_index for masked tokens)

        Returns:
            Sum of losses over all valid tokens
        """
        return reduction.apply_sum_reduction(logits, labels, self.vocab_size, self.ignore_index)


def compute_sft_loss(
    logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
    labels: jaxtyping.Int[torch.Tensor, "batch seq"],
    vocab_size: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute SFT loss with the specified reduction.

    This is a convenience function that handles both mean and sum reduction.

    Args:
        logits: Model predictions
        labels: Target labels
        vocab_size: Vocabulary size
        reduction: "mean" or "sum"

    Returns:
        Loss value
    """
    if reduction == "mean":
        loss_fn = MaskedCrossEntropyLoss(vocab_size=vocab_size, reduction="mean")
    elif reduction == "sum":
        loss_fn = SumCrossEntropyLoss(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return loss_fn(logits, labels)
