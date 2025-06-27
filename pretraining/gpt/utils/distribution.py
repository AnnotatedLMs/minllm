# Standard Library
import os
import typing

# Third Party
import torch.distributed as dist


def setup_ddp(backend: str = "nccl") -> typing.Tuple[int, int, int]:
    """
    Initialize DDP if running under torchrun.

    Returns:
        ddp_rank: Global rank of this process
        ddp_local_rank: Local rank on this node
        ddp_world_size: Total number of processes
    """
    # Check if we're running under torchrun (DDP)
    if "RANK" in os.environ:
        # DDP setup
        dist.init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        print(
            f"DDP initialized: rank={ddp_rank}, local_rank={ddp_local_rank}, world_size={ddp_world_size}"
        )
    else:
        # Single GPU setup
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1

    return ddp_rank, ddp_local_rank, ddp_world_size


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


def get_world_size() -> int:
    """Get number of distributed processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
