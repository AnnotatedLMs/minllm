# Standard Library
import typing

# Third Party
from datasets import Dataset

# Project
from posttraining.instruction_tuning.configs.data import dataset_config
from posttraining.instruction_tuning.data.transforms import sft_transforms


def calculate_token_statistics(dataset: Dataset) -> typing.Dict[str, typing.Union[int, float]]:
    """
    Calculate token statistics for a tokenized dataset.

    Args:
        dataset: Tokenized dataset

    Returns:
        Dictionary with token statistics
    """
    if sft_transforms.INPUT_IDS_KEY not in dataset.column_names:
        return {}

    total_tokens = 0
    trainable_tokens = 0

    for sample in dataset:
        tokens = len(sample[sft_transforms.INPUT_IDS_KEY])
        total_tokens += tokens

        if sft_transforms.LABELS_KEY in sample:
            trainable_tokens += sum(
                1
                for label in sample[sft_transforms.LABELS_KEY]
                if label != sft_transforms.IGNORE_INDEX
            )

    return {
        "total_tokens": total_tokens,
        "trainable_tokens": trainable_tokens,
        "avg_tokens_per_instance": (total_tokens / len(dataset) if len(dataset) > 0 else 0),
    }


def collect_dataset_statistics(
    dc: dataset_config.DatasetConfig,
    initial_size: int,
    final_dataset: Dataset,
) -> typing.Dict[str, typing.Any]:
    """
    Collect statistics for a single dataset.

    Args:
        dc: Dataset configuration
        initial_size: Size before processing
        final_dataset: Dataset after processing

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "dataset_name": dc.dataset_name,
        "dataset_split": dc.dataset_split,
        "initial_instances": initial_size,
        "final_instances": len(final_dataset),
        "instances_filtered": initial_size - len(final_dataset),
        "frac_or_num_samples": dc.frac_or_num_samples,
        "original_dataset_size": dc.original_dataset_size,
        "is_upsampled": dc.is_upsampled,
        "upsampling_factor": calculate_upsampling_factor(dc),
    }

    # Add token statistics if available
    token_stats = calculate_token_statistics(final_dataset)
    stats.update(token_stats)

    return stats


def calculate_upsampling_factor(dc: dataset_config.DatasetConfig) -> float:
    """
    Calculate upsampling factor for a dataset.

    Args:
        dc: Dataset configuration

    Returns:
        Upsampling factor
    """
    if dc.original_dataset_size and dc.original_dataset_size > 0:
        return dc.dataset_range / dc.original_dataset_size
    return 1.0


def aggregate_statistics(
    dataset_statistics: typing.List[typing.Dict[str, typing.Any]],
    dataset_order: typing.List[str],
) -> typing.Dict[str, typing.Any]:
    """
    Aggregate statistics from multiple datasets.

    Args:
        dataset_statistics: List of per-dataset statistics
        dataset_order: Order of datasets

    Returns:
        Aggregated statistics dictionary
    """
    return {
        "per_dataset_stats": dataset_statistics,
        "dataset_order": dataset_order,
    }
