# Standard Library
import typing

# Third Party
import torch

# Project
from pretraining.utils.training import metrics as metrics_module


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataloader: typing.Any,  # PretrainDataLoader
    num_batches: int = 50,
    split: str = "val",
) -> typing.Dict[str, float]:
    """Estimate loss on a data split.

    Simple evaluation for LLM pretraining - just computes average loss
    and perplexity over a fixed number of batches.

    Args:
        model: Model to evaluate
        dataloader: Data loader with get_batch method
        num_batches: Number of batches to evaluate on
        split: Data split to use ("train" or "val")

    Returns:
        Dictionary with loss and perplexity
    """
    model.eval()
    losses = []

    for _ in range(num_batches):
        inputs, targets = dataloader.get_batch(split)

        # Use inference_forward for efficiency (no intermediate activations)
        logits = model.inference_forward(inputs)

        # Compute loss
        loss = metrics_module.compute_loss(logits, targets)
        losses.append(loss.item())

    # Compute average metrics
    avg_loss = sum(losses) / len(losses)
    perplexity = metrics_module.compute_perplexity(avg_loss)

    model.train()
    return {f"{split}/loss": avg_loss, f"{split}/perplexity": perplexity}
