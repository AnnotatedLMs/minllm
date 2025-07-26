# Standard Library
import gc
import random
from typing import Optional
from typing import TypeVar

# Third Party
import numpy as np
import torch
import torch.distributed as dist

# Project
from pretraining.utils.training import distributed

T = TypeVar("T")
V = TypeVar("V", bool, int, float)


def seed_all(seed: int) -> None:
    """Seed all rng objects."""
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


def move_to_device(o: T, device: torch.device) -> T:
    """Recursively move tensors to device."""
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


def peak_gpu_memory(reset: bool = False) -> Optional[float]:
    """Get the peak GPU memory usage in MB across all ranks.

    Only rank 0 will get the final result.

    Args:
        reset: Whether to reset peak memory stats after reading

    Returns:
        Peak memory in MB (only on rank 0)
    """
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    peak_mb = torch.cuda.max_memory_allocated(device) / 1000000
    if distributed.is_distributed():
        peak_mb_tensor = torch.tensor(peak_mb, device=device)
        dist.reduce(peak_mb_tensor, 0, dist.ReduceOp.MAX)
        peak_mb = peak_mb_tensor.item()

    if reset:
        # Reset peak stats.
        torch.cuda.reset_max_memory_allocated(device)

    return peak_mb


def gc_cuda() -> None:
    """Garbage collect and empty CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_finite_(
    x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False
) -> None:
    """Modify tensor in place to replace infinities with finite values.

    Args:
        x: Tensor to modify
        check_neg_inf: Replace -inf with dtype min value
        check_pos_inf: Replace +inf with dtype max value
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)
