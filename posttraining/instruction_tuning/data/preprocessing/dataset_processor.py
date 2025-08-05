# Standard Library
import typing

# Third Party
from datasets import Dataset
from datasets import concatenate_datasets

# Project
from posttraining.instruction_tuning.configs.data import dataset_config
from posttraining.instruction_tuning.configs.data import tokenizer_config
from posttraining.instruction_tuning.data.preprocessing import dataset_transforms
from posttraining.instruction_tuning.data.transforms import sft_transforms
from posttraining.instruction_tuning.utils import cpu_utils

# Constants
DATASET_ORIGIN_KEY = "dataset_source"
TOKENIZED_SFT_DATASET_KEYS = [
    sft_transforms.INPUT_IDS_KEY,
    "attention_mask",
    sft_transforms.LABELS_KEY,
]

# Performance tuning constants
APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU = 400
FILTER_EXAMPLE_PER_SECOND_PER_CPU = 1130

# Transform function registry
TRANSFORM_FNS = {
    "sft_tokenize_v1": (sft_transforms.sft_tokenize_v1, "map"),
    "sft_tokenize_mask_out_prompt_v1": (sft_transforms.sft_tokenize_mask_out_prompt_v1, "map"),
    "sft_filter_v1": (sft_transforms.sft_filter_v1, "filter"),
    "sft_tulu_tokenize_and_truncate_v1": (sft_transforms.sft_tulu_tokenize_and_truncate_v1, "map"),
    "sft_tulu_filter_v1": (sft_transforms.sft_tulu_filter_v1, "filter"),
}


def get_num_proc(dataset_len: int, num_available_cpus: int, example_per_second_per_cpu: int) -> int:
    """
    Calculate optimal number of processes for dataset operations.

    Args:
        dataset_len: Length of the dataset
        num_available_cpus: Available CPU cores
        example_per_second_per_cpu: Processing speed per CPU

    Returns:
        Optimal number of processes
    """
    num_required_cpus = max(1, dataset_len // example_per_second_per_cpu)
    return min(num_required_cpus, num_available_cpus)


def process_dataset(
    dc: dataset_config.DatasetConfig, tc: tokenizer_config.TokenizerConfig
) -> Dataset:
    """
    Process a single dataset with transformations.

    Args:
        dc: Dataset configuration
        tc: Tokenizer configuration

    Returns:
        Processed dataset
    """
    # Validate configuration
    dataset_transforms.validate_transform_config(dc)

    # Get number of processes
    num_proc = cpu_utils.get_available_cpu_count()

    dataset = dc.dataset

    # Add dataset source field for tracking
    dataset = dataset_transforms.add_dataset_source_field(dataset, dc.dataset_name, num_proc)

    # Apply transformations
    for fn_name, fn_args in zip(dc.transform_fn, dc.transform_fn_args):
        # Determine target columns for this transform
        target_columns = dataset_transforms.determine_target_columns(dataset, dc.target_columns)

        # Apply transformation
        dataset = dataset_transforms.apply_single_transform(
            dataset=dataset,
            transform_name=fn_name,
            transform_args=fn_args,
            target_columns=target_columns,
            tokenizer_config=tc,
            num_proc=num_proc,
        )

    if len(dataset) == 0:
        raise ValueError("No examples left after transformation")

    return dataset


def remove_dataset_source_field(dataset: Dataset) -> Dataset:
    """
    Remove dataset_source field from dataset if it exists.

    This should be called after statistics collection but before returning
    the final dataset to avoid storing unnecessary metadata in cached datasets.

    Args:
        dataset: Dataset to clean

    Returns:
        Dataset without source field
    """
    if DATASET_ORIGIN_KEY in dataset.column_names:
        return dataset.remove_columns([DATASET_ORIGIN_KEY])
    return dataset


def process_and_combine_datasets(
    dcs: typing.List[dataset_config.DatasetConfig], tc: tokenizer_config.TokenizerConfig
) -> typing.Tuple[Dataset, typing.Dict[str, typing.Any]]:
    """
    Process multiple datasets and combine them.

    Args:
        dcs: List of dataset configurations
        tc: Tokenizer configuration

    Returns:
        Tuple of (combined dataset, statistics)
    """
    transformed_datasets = []
    dataset_statistics = []
    dataset_order = []

    for dc in dcs:
        # Get initial dataset info
        initial_size = len(dc.dataset) if dc.dataset else 0

        # Process dataset
        dataset = process_dataset(dc, tc)
        transformed_datasets.append(dataset)

        # Collect statistics
        stats = dataset_statistics.collect_dataset_statistics(dc, initial_size, dataset)
        dataset_statistics.append(stats)
        dataset_order.append(dc.dataset_name)

    # Combine datasets
    combined_dataset = concatenate_datasets(transformed_datasets)

    # Aggregate statistics
    all_statistics = dataset_statistics.aggregate_statistics(dataset_statistics, dataset_order)

    return combined_dataset, all_statistics
