# Standard Library
import os
import typing

# Third Party
import torch
import torch.distributed as dist
import torch.nn


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1


def get_global_rank() -> int:
    """Get global rank of current process."""
    if is_distributed():
        return int(os.environ.get("RANK") or dist.get_rank())
    else:
        return 0


def get_local_rank() -> int:
    """Get local rank on current node."""
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
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def synchronize_value(
    value: typing.Union[bool, int, float], device: torch.device
) -> typing.Union[bool, int, float]:
    """Synchronize a value across all ranks by broadcasting from rank 0."""
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
    """Clean up DDP process group."""
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
