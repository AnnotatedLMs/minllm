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
    """Initialize process group for DDP (Distributed Data Parallel) training.

    What This Does (for ML researchers new to distributed training):

    DDP enables training a single model across multiple GPUs/machines by:
    1. Creating identical copies of your model on each GPU
    2. Splitting your batch across GPUs (each GPU gets batch_size/n_gpus)
    3. Computing forward/backward passes independently on each GPU
    4. Synchronizing gradients across all GPUs before optimizer step
    5. Ensuring all model copies stay in sync

    The "process group" is the communication layer that enables GPUs to:
    - Share gradients during backprop (all-reduce operation)
    - Broadcast model parameters from rank 0 at initialization
    - Synchronize at barriers to prevent race conditions

    Backend Selection:
    - NCCL (NVIDIA Collective Communications Library): Optimized for GPU-to-GPU
      communication. Uses high-bandwidth GPU interconnects (NVLink, InfiniBand)
    - Gloo: CPU-based fallback for non-CUDA environments

    LOCAL_RANK Magic:
    - Each GPU on a machine gets a local rank (0, 1, 2, 3 for 4 GPUs)
    - This sets CUDA device to match LOCAL_RANK, ensuring each process
      uses exactly one GPU and prevents conflicts

    Should only be called when RANK is set in environment (i.e., under torchrun).

    Args:
        timeout_minutes: How long to wait for all processes to join before failing.
                        Important for debugging - if one process crashes, others
                        will wait this long before timing out.
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        local_rank = distributed.get_local_rank()
        torch.cuda.set_device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(minutes=timeout_minutes),
    )
