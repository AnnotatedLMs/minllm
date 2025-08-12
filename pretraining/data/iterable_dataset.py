# Standard Library
import math
import pathlib
import typing

# Third Party
import numpy as np
from torch.utils import data

# Project
from pretraining.utils.training import dist_utils


class IterableDataset(data.IterableDataset[typing.Dict[str, typing.Any]]):
    """
    Wraps an indexable dataset as an IterableDataset for distributed training.

    Features:
    - Deterministic data ordering with resumption support
    - Proper sharding across distributed ranks
    - Worker-aware sharding for DataLoader workers
    - Checkpoint/resume via start_index

    Args:
        dataset: Underlying indexable dataset
        global_batch_size: Total batch size across all GPUs
        seed: Random seed for shuffling
        epoch: Current epoch (affects shuffling)
        start_index: Resume training from this global index
        drop_last: Drop incomplete batches
        work_dir: Directory to save global indices for reproducibility
    """

    def __init__(
        self,
        dataset: data.Dataset,
        global_batch_size: int,
        seed: int = 0,
        epoch: int = 0,
        start_index: int = 0,
        drop_last: bool = True,
        work_dir: typing.Optional[pathlib.Path] = None,
    ):
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.seed = seed
        self.epoch = epoch
        self.start_index = start_index
        self.drop_last = drop_last
        self.work_dir = work_dir

        # Get distributed info
        self.rank = dist_utils.get_global_rank()
        self.world_size = dist_utils.get_world_size()

        # Validate and calculate sizes
        self._validate_batch_size()
        self._calculate_sizes()

        # Save/load global indices for reproducibility
        self._setup_global_indices()

    def _validate_batch_size(self) -> None:
        """Ensure global batch size is divisible by world size."""
        if self.global_batch_size % self.world_size != 0:
            raise ValueError(
                f"global_batch_size ({self.global_batch_size}) must be divisible "
                f"by world_size ({self.world_size})"
            )

    def _calculate_sizes(self) -> None:
        """Calculate per-device batch size and total dataset size."""
        self.device_batch_size = self.global_batch_size // self.world_size

        if self.drop_last and len(self.dataset) % self.world_size != 0:
            # Round down to ensure all ranks get same amount
            num_samples = (len(self.dataset) // self.world_size) * self.world_size
        else:
            # Round up and pad if needed
            num_samples = math.ceil(len(self.dataset) / self.world_size) * self.world_size

        self.total_size = num_samples

    def _setup_global_indices(self) -> None:
        """Initialize global indices file path if using work_dir."""
        self.global_indices_file = None
        if self.work_dir is not None:
            self.global_indices_file = self.work_dir / f"global_indices_epoch{self.epoch}.npy"
            self._save_or_load_global_indices()

    def _save_or_load_global_indices(self) -> None:
        """Save global indices on rank 0, load on all ranks."""
        if self.rank == 0:
            self._save_global_indices()
        dist_utils.barrier()  # Wait for rank 0 to save

    def _save_global_indices(self) -> None:
        """Build and save global indices to file."""
        if self.global_indices_file is None:
            return

        self.global_indices_file.parent.mkdir(parents=True, exist_ok=True)
        indices = self._build_global_indices()

        # Save as memory-mapped array
        indices_mmap = np.memmap(
            self.global_indices_file, dtype=np.int64, mode="w+", shape=(len(indices),)
        )
        indices_mmap[:] = indices
        indices_mmap.flush()
        del indices_mmap

    def _build_global_indices(self) -> np.ndarray:
        """Build shuffled or sequential indices for entire dataset."""
        indices = np.arange(len(self.dataset), dtype=np.int64)

        if self._should_shuffle():
            indices = self._shuffle_indices(indices)

        if not self.drop_last:
            indices = self._pad_indices(indices)
        else:
            indices = self._truncate_indices(indices)

        return indices

    def _should_shuffle(self) -> bool:
        """Determine if we should shuffle (typically for training)."""
        # Could be made configurable
        return True

    def _shuffle_indices(self, indices: np.ndarray) -> np.ndarray:
        """Deterministically shuffle indices based on seed and epoch."""
        rng = np.random.Generator(np.random.PCG64(seed=self.seed + self.epoch))
        rng.shuffle(indices)
        return indices

    def _pad_indices(self, indices: np.ndarray) -> np.ndarray:
        """Pad indices to make divisible by world_size."""
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            # Repeat indices from beginning
            padding = indices[:padding_size]
            indices = np.concatenate([indices, padding])
        return indices

    def _truncate_indices(self, indices: np.ndarray) -> np.ndarray:
        """Truncate indices to make divisible by world_size."""
        return indices[: self.total_size]

    def get_global_indices(self) -> np.ndarray:
        """Load or build global indices."""
        if self.global_indices_file is not None and self.global_indices_file.exists():
            return np.memmap(self.global_indices_file, mode="r", dtype=np.int64)
        else:
            return self._build_global_indices()

    def __iter__(self) -> typing.Iterator[typing.Dict[str, typing.Any]]:
        """Iterate over dataset with proper distributed sharding."""
        indices = self.get_global_indices()

        # Apply start_index for resumption
        if self.start_index > 0:
            indices = indices[self.start_index :]

        # Shard by distributed rank
        rank_indices = self._shard_indices_by_rank(indices)

        # Further shard by dataloader workers if needed
        worker_indices = self._shard_indices_by_worker(rank_indices)

        # Yield data for each index
        for idx in worker_indices:
            yield self.dataset[int(idx)]

    def _shard_indices_by_rank(self, indices: np.ndarray) -> np.ndarray:
        """Shard indices across distributed ranks."""
        # Each rank takes every world_size-th element starting from rank
        return indices[self.rank :: self.world_size]

    def _shard_indices_by_worker(self, indices: np.ndarray) -> np.ndarray:
        """Further shard indices across DataLoader workers."""
        worker_info = data.get_worker_info()

        if worker_info is None:
            # Single worker - return all indices
            return indices

        # Shard across workers
        # Each worker handles chunks of device_batch_size to maintain order
        num_workers = worker_info.num_workers
        worker_id = worker_info.id

        # Reshape to chunks, distribute chunks, then flatten
        num_complete_batches = len(indices) // self.device_batch_size
        complete_size = num_complete_batches * self.device_batch_size

        # Handle complete batches
        if complete_size > 0:
            complete_indices = indices[:complete_size].reshape(-1, self.device_batch_size)
            worker_batches = complete_indices[worker_id::num_workers]
            worker_indices = worker_batches.reshape(-1)
        else:
            worker_indices = np.array([], dtype=indices.dtype)

        # Handle remainder
        if complete_size < len(indices):
            remainder = indices[complete_size:]
            worker_remainder = remainder[worker_id::num_workers]
            worker_indices = np.concatenate([worker_indices, worker_remainder])

        return worker_indices

    def set_epoch(self, epoch: int) -> None:
        """Update epoch for different shuffling."""
        self.epoch = epoch
        if self.work_dir is not None:
            self._setup_global_indices()
