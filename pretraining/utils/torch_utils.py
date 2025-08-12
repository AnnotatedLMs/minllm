# Standard Library
import gc
import random
import typing

# Third Party
import numpy as np
import torch
from torch import distributed as dist

# Project
from pretraining.utils.training import dist_utils

T = typing.TypeVar("T")
V = typing.TypeVar("V", bool, int, float)


def seed_all(seed: int) -> None:
    """Seed all random number generators for reproducible training.

    Why This Matters for ML Research:
    - Ensures experiments are reproducible (same seed = same results)
    - Critical for debugging (can replay exact failure conditions)
    - Required for scientific validity when comparing methods

    What Gets Seeded:
    1. Python's random module (data shuffling, augmentations)
    2. NumPy (array operations, data preprocessing)
    3. PyTorch CPU (model initialization, dropout)
    4. PyTorch CUDA (GPU operations, cuDNN)

    Note: Even with seeding, some operations (like atomicAdd on GPU)
    may still be non-deterministic. For full determinism, you may need:
    - torch.use_deterministic_algorithms(True)
    - Set CUBLAS_WORKSPACE_CONFIG environment variable
    """
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


def move_to_device(o: T, device: torch.device) -> T:
    """Recursively move tensors to device (CPU/GPU/MPS).

    Why Recursive Movement is Needed:
    - Model outputs often contain nested structures (dicts of tensors)
    - Batch data might be tuples/lists of tensors

    Common use case: batch = move_to_device(batch, 'cuda')
    """
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o


def get_default_device() -> torch.device:
    """Get default device based on availability."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def peak_gpu_memory(reset: bool = False) -> typing.Optional[float]:
    """Get the peak GPU memory usage in MB across all distributed ranks.

    How GPU Memory Tracking Works:
    - PyTorch tracks peak allocated memory since last reset
    - Different from nvidia-smi (includes all CUDA context overhead)
    - Only tracks tensors/buffers allocated by PyTorch

    In Distributed Training:
    - Each GPU might have different peak usage (data imbalance)
    - This function finds the MAX across all GPUs
    - Critical for understanding if one GPU is bottlenecking

    Memory Debugging Tips:
    - Call with reset=True at start of training loop
    - Check after forward pass, after backward, after optimizer
    - Large jumps indicate memory leaks or unexpected allocations

    Args:
        reset: Whether to reset peak memory stats after reading

    Returns:
        Peak memory in MB (only on rank 0, None on other ranks in distributed)
    """
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    peak_mb = torch.cuda.max_memory_allocated(device) / 1000000
    if dist_utils.is_distributed():
        peak_mb_tensor = torch.tensor(peak_mb, device=device)
        dist.reduce(peak_mb_tensor, 0, dist.ReduceOp.MAX)
        peak_mb = peak_mb_tensor.item()

    if reset:
        # Reset peak stats.
        torch.cuda.reset_max_memory_allocated(device)

    return peak_mb


def gc_cuda() -> None:
    """Garbage collect Python objects and empty CUDA memory cache.

    Why This Helps:
    - PyTorch caches GPU memory for faster allocation (doesn't return to OS)
    - Dead tensors might still occupy cache until Python GC runs
    - Can prevent OOM errors when switching between models

    What Happens:
    1. gc.collect(): Forces Python garbage collection, deleting dead objects
    2. empty_cache(): Returns cached GPU memory to PyTorch's allocator
       (Note: Doesn't reduce nvidia-smi memory usage, but makes it reusable)

    When to Use:
    - Between training different model configurations
    - After deleting large models/optimizers
    - When debugging GPU memory issues
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_finite_(
    x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False
) -> None:
    """Modify tensor in place to replace infinities with finite values.

    Why Infinities Appear in ML:
    - Log of zero produces -inf (common in log probabilities)
    - Division by zero in attention scores (before masking)
    - Overflow in exp() operations (attention without scaling)
    - Numerical instabilities in deep networks

    What This Does:
    - Replaces -inf with smallest representable value for dtype
    - Replaces +inf with largest representable value for dtype
    - Prevents NaN propagation in subsequent operations

    Common Use Case:
    - After computing log probabilities: ensure_finite_(log_probs)
    - Before softmax on masked attention: ensure_finite_(scores)

    Args:
        x: Tensor to modify in-place
        check_neg_inf: Replace -inf with dtype min value
        check_pos_inf: Replace +inf with dtype max value
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)
