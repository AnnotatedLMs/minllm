# Standard Library
import typing

# Project
from posttraining.instruction_tuning.configs.data import dataset_config
from posttraining.instruction_tuning.data.utils import loading
from posttraining.instruction_tuning.data.utils import sampling


def parse_dataset_mixer_list(
    dataset_mixer_list: typing.List[str],
    dataset_mixer_list_splits: typing.List[str],
    dataset_transform_fn: typing.List[str],
    transform_fn_args: typing.List[typing.Dict[str, typing.Any]],
    target_columns: typing.Optional[typing.List[str]] = None,
) -> typing.List[dataset_config.DatasetConfig]:
    """
    Parse dataset mixer list into DatasetConfig objects.

    Args:
        dataset_mixer_list: List of [dataset_name, weight, dataset_name, weight, ...]
        dataset_mixer_list_splits: List of splits for each dataset
        dataset_transform_fn: Transform functions to apply
        transform_fn_args: Arguments for transform functions
        target_columns: Columns to keep after transformation

    Returns:
        List of DatasetConfig objects
    """
    # Validate format
    loading.validate_mixer_list_format(dataset_mixer_list)

    # Handle splits
    splits = loading.validate_dataset_splits(dataset_mixer_list, dataset_mixer_list_splits)

    # Parse dataset configs
    dataset_configs = []
    for i in range(0, len(dataset_mixer_list), 2):
        dataset_name = dataset_mixer_list[i]
        weight_str = dataset_mixer_list[i + 1]

        # Parse weight
        frac_or_num_samples = sampling.parse_weight_specification(weight_str)

        # Create dataset config
        dataset_config = dataset_config.DatasetConfig(
            dataset_name=dataset_name,
            dataset_split=splits[i // 2],
            dataset_revision="main",
            transform_fn=dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=target_columns,
            frac_or_num_samples=frac_or_num_samples,
        )

        # Calculate and set target range
        original_size = len(dataset_config.dataset)
        new_range = sampling.calculate_dataset_range(frac_or_num_samples, original_size)

        print(
            f"Dataset {dataset_name}: {original_size} -> {new_range} samples "
            f"(factor: {frac_or_num_samples})"
        )

        dataset_config.update_range(new_range)
        dataset_configs.append(dataset_config)

    return dataset_configs


def get_dataset_stats(
    dataset_configs: typing.List[dataset_config.DatasetConfig],
) -> typing.Dict[str, typing.Any]:
    """
    Get statistics about dataset mixture.

    Args:
        dataset_configs: List of dataset configurations

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "total_samples": sum(dc.dataset_range for dc in dataset_configs),
        "num_datasets": len(dataset_configs),
        "datasets": [],
    }

    for dc in dataset_configs:
        dataset_info = {
            "name": dc.dataset_name,
            "split": dc.dataset_split,
            "original_size": dc.original_dataset_size,
            "target_size": dc.dataset_range,
            "is_upsampled": dc.is_upsampled,
            "weight": dc.frac_or_num_samples,
        }
        stats["datasets"].append(dataset_info)

    return stats
