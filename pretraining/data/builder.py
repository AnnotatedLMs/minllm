# Standard Library
import pathlib
import typing

# Third Party
import torch
import torch.utils.data

# Project
from pretraining.configs.model.architectures import base as model_base
from pretraining.configs.training import data_configs
from pretraining.configs.training import trainer_configs
from pretraining.data import collator as data_collator
from pretraining.data import iterable_dataset
from pretraining.data import memmap_dataset
from pretraining.utils.training import distributed


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
    data_path = data_dir / f"{split}.bin"

    # For now, single file per split
    # Could be extended to multiple files
    dataset = memmap_dataset.MemMapDataset(
        data_path,
        chunk_size=chunk_size,
        metadata=[{"split": split, "source": str(data_path)}],
    )

    return dataset


def build_train_dataloader(
    training_config: trainer_configs.TrainingLoopConfig,
    model_config: model_base.BaseLLMConfig,
    start_index: int = 0,
    epoch: int = 0,
) -> torch.utils.data.DataLoader:
    """Build training dataloader with all components.

    Args:
        training_config: Full training configuration
        model_config: Model configuration (for collator selection)
        start_index: Resume from this global index
        epoch: Current epoch for shuffling

    Returns:
        PyTorch DataLoader ready for training
    """
    # Build base dataset
    base_dataset = build_memmap_dataset(
        training_config.data,
        split="train",
        chunk_size=training_config.batch.sequence_length,
    )

    # Wrap in iterable dataset for distributed training
    dataset = build_iterable_dataset(
        base_dataset,
        training_config,
        start_index=start_index,
        epoch=epoch,
        drop_last=True,
    )

    # Build appropriate collator
    collator = data_collator.build_collator(model_config)

    # Create DataLoader
    dataloader = build_dataloader(
        dataset,
        training_config,
        collator=collator,
        is_train=True,
    )

    return dataloader


def build_eval_dataloader(
    training_config: trainer_configs.TrainingLoopConfig,
    model_config: model_base.BaseLLMConfig,
    split: typing.Literal["val", "test"] = "val",
) -> torch.utils.data.DataLoader:
    """Build evaluation dataloader.

    Args:
        training_config: Full training configuration
        model_config: Model configuration (for collator selection)
        split: Evaluation split

    Returns:
        PyTorch DataLoader for evaluation
    """
    # Build base dataset
    base_dataset = build_memmap_dataset(
        training_config.data,
        split=split,
        chunk_size=training_config.batch.sequence_length,
    )

    # For eval, we don't shuffle and use DistributedSampler directly
    sampler = build_distributed_sampler(
        base_dataset,
        shuffle=False,
        drop_last=False,
    )

    # Use standard collator for eval (no MTP needed)
    collator = data_collator.DataCollator()

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        base_dataset,
        batch_size=training_config.batch.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=training_config.data.num_workers,
        pin_memory=training_config.data.pin_memory,
        drop_last=False,
    )

    return dataloader


def build_iterable_dataset(
    base_dataset: torch.utils.data.Dataset,
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
    global_batch_size = training_config.batch.batch_size * distributed.get_world_size()

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
    dataset: torch.utils.data.Dataset,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: int = 0,
) -> torch.utils.data.DistributedSampler:
    """Build distributed sampler for evaluation.

    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle
        drop_last: Whether to drop last incomplete batch
        seed: Random seed

    Returns:
        DistributedSampler instance
    """
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=distributed.get_world_size(),
        rank=distributed.get_global_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )

    return sampler


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    training_config: trainer_configs.TrainingLoopConfig,
    collator: data_collator.DataCollator,
    is_train: bool = True,
) -> torch.utils.data.DataLoader:
    """Build PyTorch DataLoader with proper settings.

    Args:
        dataset: Dataset to load from
        training_config: Training configuration
        collator: Data collator for batching
        is_train: Whether this is for training

    Returns:
        Configured DataLoader
    """
    # For IterableDataset, we don't use a sampler
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_config.batch.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=training_config.data.num_workers,
        pin_memory=training_config.data.pin_memory,
        drop_last=is_train,  # Drop last for training, keep for eval
        persistent_workers=training_config.data.num_workers > 0,
    )

    return dataloader
