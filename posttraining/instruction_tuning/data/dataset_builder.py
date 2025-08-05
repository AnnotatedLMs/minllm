# Standard Library
import typing
from typing import Literal

# Third Party
from datasets import Dataset

# Project
from posttraining.instruction_tuning.configs.data import tokenizer_config
from posttraining.instruction_tuning.data.preprocessing import cache_manager
from posttraining.instruction_tuning.data.preprocessing import dataset_processor
from posttraining.instruction_tuning.data.utils import mixing


def get_cached_sft_dataset(
    dataset_mixer_list: typing.List[str],
    dataset_mixer_list_splits: typing.List[str],
    tc: tokenizer_config.TokenizerConfig,
    dataset_transform_fn: typing.List[str],
    transform_fn_args: typing.List[typing.Dict[str, typing.Any]],
    target_columns: typing.Optional[typing.List[str]] = None,
    dataset_cache_mode: Literal["hf", "local"] = "local",
    dataset_config_hash: typing.Optional[str] = None,
    hf_entity: typing.Optional[str] = None,
    dataset_local_cache_dir: str = "local_dataset_cache",
    dataset_skip_cache: bool = False,
    drop_dataset_source: bool = True,
) -> typing.Tuple[Dataset, typing.Dict[str, typing.Any]]:
    """
    Get a cached SFT dataset with statistics.

    This is the main entry point for loading and processing SFT datasets.
    It handles:
    1. Parsing dataset configurations
    2. Checking cache
    3. Processing datasets if not cached
    4. Saving to cache
    5. Returning dataset with statistics

    Args:
        dataset_mixer_list: List of [dataset_name, weight, ...] pairs
        dataset_mixer_list_splits: Splits for each dataset
        tc: Tokenizer configuration
        dataset_transform_fn: Transform functions to apply
        transform_fn_args: Arguments for transform functions
        target_columns: Columns to keep after transformation
        dataset_cache_mode: "local" or "hf" caching
        dataset_config_hash: Optional pre-computed hash
        hf_entity: HuggingFace entity for HF caching
        dataset_local_cache_dir: Directory for local caching
        dataset_skip_cache: Whether to skip cache
        drop_dataset_source: Whether to remove source tracking field

    Returns:
        Tuple of (processed dataset, statistics dictionary)
    """
    # Parse dataset configurations
    dcs = []
    if dataset_config_hash is None:
        dcs = mixing.parse_dataset_mixer_list(
            dataset_mixer_list,
            dataset_mixer_list_splits,
            dataset_transform_fn,
            transform_fn_args,
            target_columns,
        )
        dataset_config_hash = cache_manager.compute_config_hash(dcs, tc)

    # Create cache manager
    if dataset_cache_mode == "local":
        cache = cache_manager.LocalDatasetTransformationCache(
            config_hash=dataset_config_hash, dataset_local_cache_dir=dataset_local_cache_dir
        )
    elif dataset_cache_mode == "hf":
        cache = cache_manager.DatasetTransformationCache(
            config_hash=dataset_config_hash, hf_entity=hf_entity
        )
    else:
        raise ValueError(f"Unknown cache mode: {dataset_cache_mode}")

    # Load or transform dataset
    dataset, statistics = cache.load_or_transform_dataset(
        dcs, tc, dataset_processor.process_and_combine_datasets, dataset_skip_cache
    )

    # Remove source field if requested
    if drop_dataset_source:
        dataset = dataset_processor.remove_dataset_source_field(dataset)

    return dataset, statistics
