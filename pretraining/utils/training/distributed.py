# Standard Library
import os
import typing

# Third Party
import torch
import torch.distributed as dist
import torch.nn


def is_distributed() -> bool:
    """Check if distributed training is initialized.

    Returns True when running under torchrun/torch.distributed.launch,
    False when running single-GPU or CPU training.
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get total number of processes across all machines.

    In distributed training context:
    - Single machine, 4 GPUs: world_size = 4
    - 2 machines, 8 GPUs each: world_size = 16
    - Single GPU/CPU: world_size = 1

    Used to calculate global batch size and learning rate scaling.
    """
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1


def get_global_rank() -> int:
    """Get global rank of current process across all machines.

    Rank is a unique ID for each process:
    - Rank 0: Master process (handles logging, saving checkpoints)
    - Rank 1-N: Worker processes

    Example with 2 machines, 4 GPUs each:
    - Machine 0: ranks 0, 1, 2, 3
    - Machine 1: ranks 4, 5, 6, 7
    """
    if is_distributed():
        return int(os.environ.get("RANK") or dist.get_rank())
    else:
        return 0


def get_local_rank() -> int:
    """Get local rank on current node/machine.

    Local rank identifies GPU within a single machine:
    - Local rank 0: First GPU on this machine (cuda:0)
    - Local rank 1: Second GPU on this machine (cuda:1)

    Critical for setting CUDA device to prevent multiple processes
    from competing for the same GPU.
    """
    return int(os.environ.get("LOCAL_RANK") or 0)


def get_local_world_size() -> int:
    """Get number of processes on current node."""
    return int(os.environ.get("LOCAL_WORLD_SIZE") or 1)


def get_node_rank() -> int:
    """Get rank of current node."""
    return int(
        os.environ.get("NODE_RANK")
        or (get_global_rank() - get_local_rank()) // get_local_world_size()
    )


def barrier() -> None:
    """Synchronize all processes - wait until all reach this point.

    Use cases:
    - After rank 0 downloads data, before other ranks access it
    - Before/after timing measurements to ensure fair comparison
    - After model initialization before training starts

    Without barriers, faster processes might start training while
    slower ones are still loading, causing hangs or crashes.
    """
    if is_distributed():
        dist.barrier()


def synchronize_value(
    value: typing.Union[bool, int, float], device: torch.device
) -> typing.Union[bool, int, float]:
    """Synchronize a value across all ranks by broadcasting from rank 0.

    Why this matters:
    - Only rank 0 might compute certain values (e.g., validation loss)
    - All ranks need the same value for consistent behavior
    - Example: early stopping decision must be synchronized or some
      ranks will stop while others continue, causing deadlock

    The broadcast operation sends rank 0's value to all other ranks.
    """
    if dist.is_available() and dist.is_initialized():
        value_tensor = torch.tensor(value, device=device)
        dist.broadcast(value_tensor, 0)
        return value_tensor.item()  # type: ignore
    else:
        return value


def synchronize_flag(flag: bool, device: torch.device) -> bool:
    """Synchronize a boolean flag across all ranks."""
    return synchronize_value(flag, device)  # type: ignore


def cleanup_ddp() -> None:
    """Clean up DDP process group and free communication resources.

    Should be called at the end of training to:
    - Release NCCL/Gloo communication buffers
    - Close network connections between processes
    - Prevent hangs in subsequent runs

    Especially important when running multiple experiments in sequence.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).
    Used to ensure only one process handles logging, checkpointing, etc.
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class SingleAccelerator(torch.nn.Module):
    """Wrapper for single-device training to provide consistent interface with DDP.

    This matches OLMo's pattern where the trainer always receives a wrapped model,
    whether it's DDP-wrapped or single-device.
    """

    process_group = None

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
