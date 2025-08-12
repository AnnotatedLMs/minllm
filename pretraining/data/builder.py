# Standard Library
import pathlib
import typing

# Third Party
from torch.utils import data

# Project
from pretraining.configs.training import data_configs
from pretraining.configs.training import trainer_configs
from pretraining.data import iterable_dataset
from pretraining.data import memmap_dataset
from pretraining.utils.training import dist_utils


def build_memmap_dataset(
    data_config: data_configs.DataConfig,
    split: typing.Literal["train", "val"],
    chunk_size: int,
) -> memmap_dataset.MemMapDataset:
    """Build memory-mapped dataset for a specific split.

    Args:
        data_config: Data configuration
        split: Dataset split ("train" or "val")
        chunk_size: Number of tokens per chunk (typically max_sequence_length)

    Returns:
        MemMapDataset instance
    """
    data_dir = pathlib.Path(data_config.data_dir)

    # Check for FineWeb-style multiple files first
    if split == "train":
        pattern = "fineweb_train_*.bin"
    else:
        pattern = "fineweb_val_*.bin"

    fineweb_files = sorted(data_dir.glob(pattern))

    if fineweb_files:
        # Use FineWeb files if they exist
        dataset = memmap_dataset.MemMapDataset(
            *fineweb_files,
            chunk_size=chunk_size,
            metadata=[{"split": split, "source": str(f)} for f in fineweb_files],
        )
    else:
        # Fall back to simple train.bin/val.bin structure
        data_path = data_dir / f"{split}.bin"
        dataset = memmap_dataset.MemMapDataset(
            data_path,
            chunk_size=chunk_size,
            metadata=[{"split": split, "source": str(data_path)}],
        )

    return dataset


def build_iterable_dataset(
    base_dataset: data.Dataset,
    training_config: trainer_configs.TrainingLoopConfig,
    start_index: int = 0,
    epoch: int = 0,
    drop_last: bool = True,
) -> iterable_dataset.IterableDataset:
    """Wrap base dataset in IterableDataset for distributed training.

    Args:
        base_dataset: Underlying indexable dataset
        training_config: Training configuration
        start_index: Resume from this index
        epoch: Current epoch
        drop_last: Whether to drop incomplete batches

    Returns:
        IterableDataset instance
    """
    global_batch_size = training_config.batch.batch_size * dist_utils.get_world_size()

    # Create work directory for saving indices
    work_dir = None
    if training_config.checkpoint.save_dir:
        work_dir = pathlib.Path(training_config.checkpoint.save_dir) / "data_indices"

    dataset = iterable_dataset.IterableDataset(
        base_dataset,
        global_batch_size=global_batch_size,
        seed=training_config.seed,
        epoch=epoch,
        start_index=start_index,
        drop_last=drop_last,
        work_dir=work_dir,
    )

    return dataset


def build_distributed_sampler(
    dataset: data.Dataset,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: int = 0,
) -> data.DistributedSampler:
    """Build distributed sampler for evaluation.

    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle
        drop_last: Whether to drop last incomplete batch
        seed: Random seed

    Returns:
        DistributedSampler instance
    """
    sampler = data.DistributedSampler(
        dataset,
        num_replicas=dist_utils.get_world_size(),
        rank=dist_utils.get_global_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )

    return sampler
