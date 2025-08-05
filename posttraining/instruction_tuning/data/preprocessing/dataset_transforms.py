# Standard Library
import typing

# Third Party
from datasets import Dataset

# Project
from posttraining.instruction_tuning.configs.data import dataset_config
from posttraining.instruction_tuning.configs.data import tokenizer_config
from posttraining.instruction_tuning.data.preprocessing import dataset_processor


def add_dataset_source_field(
    dataset: Dataset,
    dataset_name: str,
    num_proc: int,
) -> Dataset:
    """
    Add dataset source field for tracking origin.

    Args:
        dataset: Input dataset
        dataset_name: Name of the dataset
        num_proc: Number of processes to use

    Returns:
        Dataset with source field added
    """
    return dataset.map(
        lambda example: {**example, dataset_processor.DATASET_ORIGIN_KEY: dataset_name},
        num_proc=num_proc,
        desc=f"Adding dataset source field for {dataset_name}",
    )


def prepare_transform_kwargs(
    transform_fn_name: str,
    transform_fn_args: typing.Dict[str, typing.Any],
    tokenizer_config: tokenizer_config.TokenizerConfig,
) -> typing.Dict[str, typing.Any]:
    """
    Prepare keyword arguments for transform function.

    Args:
        transform_fn_name: Name of transform function
        transform_fn_args: Arguments for transform function
        tokenizer_config: Tokenizer configuration

    Returns:
        Prepared kwargs dictionary
    """
    kwargs = {"tokenizer": tokenizer_config.tokenizer}
    kwargs.update(transform_fn_args)
    return kwargs


def determine_target_columns(
    dataset: Dataset,
    target_columns: typing.Optional[typing.List[str]],
) -> typing.List[str]:
    """
    Determine target columns to keep after transformation.

    Args:
        dataset: Input dataset
        target_columns: Specified target columns (None means keep all)

    Returns:
        List of columns to keep
    """
    # Use all columns if not specified
    columns = dataset.column_names if target_columns is None else target_columns

    # Always preserve dataset_source if it exists
    if (
        dataset_processor.DATASET_ORIGIN_KEY in dataset.column_names
        and dataset_processor.DATASET_ORIGIN_KEY not in columns
    ):
        columns = columns + [dataset_processor.DATASET_ORIGIN_KEY]

    return columns


def apply_map_transform(
    dataset: Dataset,
    transform_fn: typing.Callable,
    transform_kwargs: typing.Dict[str, typing.Any],
    target_columns: typing.List[str],
    num_proc: int,
) -> Dataset:
    """
    Apply a map transformation to dataset.

    Args:
        dataset: Input dataset
        transform_fn: Transform function to apply
        transform_kwargs: Kwargs for transform function
        target_columns: Columns to keep
        num_proc: Number of processes

    Returns:
        Transformed dataset
    """
    columns_to_remove = [col for col in dataset.column_names if col not in target_columns]

    return dataset.map(
        transform_fn,
        fn_kwargs=transform_kwargs,
        remove_columns=columns_to_remove,
        num_proc=dataset_processor.get_num_proc(
            len(dataset), num_proc, dataset_processor.APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU
        ),
    )


def apply_filter_transform(
    dataset: Dataset,
    transform_fn: typing.Callable,
    transform_kwargs: typing.Dict[str, typing.Any],
    num_proc: int,
) -> Dataset:
    """
    Apply a filter transformation to dataset.

    Args:
        dataset: Input dataset
        transform_fn: Filter function to apply
        transform_kwargs: Kwargs for filter function
        num_proc: Number of processes

    Returns:
        Filtered dataset
    """
    return dataset.filter(
        transform_fn,
        fn_kwargs=transform_kwargs,
        num_proc=dataset_processor.get_num_proc(
            len(dataset), num_proc, dataset_processor.FILTER_EXAMPLE_PER_SECOND_PER_CPU
        ),
    )


def apply_single_transform(
    dataset: Dataset,
    transform_name: str,
    transform_args: typing.Dict[str, typing.Any],
    target_columns: typing.List[str],
    tokenizer_config: tokenizer_config.TokenizerConfig,
    num_proc: int,
) -> Dataset:
    """
    Apply a single transformation to dataset.

    Args:
        dataset: Input dataset
        transform_name: Name of transform function
        transform_args: Arguments for transform
        target_columns: Columns to keep
        tokenizer_config: Tokenizer configuration
        num_proc: Number of processes

    Returns:
        Transformed dataset

    Raises:
        ValueError: If transform type is unknown
    """
    # Get transform function and type
    transform_fn, transform_type = dataset_processor.TRANSFORM_FNS[transform_name]

    # Prepare kwargs
    transform_kwargs = prepare_transform_kwargs(transform_name, transform_args, tokenizer_config)

    # Apply transformation based on type
    if transform_type == "map":
        return apply_map_transform(
            dataset,
            transform_fn,
            transform_kwargs,
            target_columns,
            num_proc,
        )
    elif transform_type == "filter":
        return apply_filter_transform(
            dataset,
            transform_fn,
            transform_kwargs,
            num_proc,
        )
    else:
        raise ValueError(f"Unknown transform function type: {transform_type}")


def validate_transform_config(dc: dataset_config.DatasetConfig) -> None:
    """
    Validate transform function configuration.

    Args:
        dc: Dataset configuration to validate

    Raises:
        AssertionError: If configuration is invalid
    """
    assert len(dc.transform_fn) == len(dc.transform_fn_args), (
        f"transform_fn and transform_fn_args must have the same length: "
        f"{dc.transform_fn=} != {dc.transform_fn_args=}"
    )
