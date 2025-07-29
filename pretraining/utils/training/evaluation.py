# Standard Library
import typing

# Third Party
import torch

# Project
from pretraining.utils.training import metrics as metrics_module


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataloader: typing.Optional[torch.utils.data.DataLoader],
    num_batches: int = 50,
) -> typing.Dict[str, float]:
    """Estimate loss on a data split.

    Simple evaluation for LLM pretraining - just computes average loss
    and perplexity over a fixed number of batches.

    Args:
        model: Model to evaluate
        dataloader: PyTorch DataLoader for evaluation
        num_batches: Number of batches to evaluate on

    Returns:
        Dictionary with loss and perplexity
    """
    if dataloader is None:
        return {"val/loss": float("inf"), "val/perplexity": float("inf")}

    model.eval()
    losses = []

    # Create iterator
    data_iter = iter(dataloader)

    for i in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            # If we run out of data, break early
            if i == 0:
                raise ValueError("Validation dataloader is empty")
            break

        # Move batch to device
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}

        inputs = batch["input_ids"]
        targets = batch["labels"]

        # Use inference_forward for efficiency (no intermediate activations)
        logits = model.inference_forward(inputs)

        # Compute loss
        loss = metrics_module.compute_loss(logits, targets)
        losses.append(loss.item())

    # Compute average metrics
    avg_loss = sum(losses) / len(losses)
    perplexity = metrics_module.compute_perplexity(avg_loss)

    model.train()
    return {"val/loss": avg_loss, "val/perplexity": perplexity}
