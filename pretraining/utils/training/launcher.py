# Standard Library
import datetime

# Third Party
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Project
from pretraining.utils.training import distributed


def setup_multiprocessing() -> None:
    """Setup multiprocessing with spawn method.

    This is required for CUDA multiprocessing to work properly.
    Safe to call multiple times - will not error if already set.
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, that's fine
        pass


def init_ddp_process_group(timeout_minutes: int = 30) -> None:
    """Initialize process group for DDP training.

    This function:
    - Detects the appropriate backend (nccl for CUDA, gloo otherwise)
    - Sets the CUDA device based on LOCAL_RANK
    - Initializes the process group

    Should only be called when RANK is set in environment (i.e., under torchrun).

    Args:
        timeout_minutes: Timeout for distributed operations
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        local_rank = distributed.get_local_rank()
        torch.cuda.set_device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(minutes=timeout_minutes),
    )
