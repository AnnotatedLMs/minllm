# Standard Library
import os
import typing

# Third Party
from datasets import Dataset
from datasets import load_dataset

# Project
from posttraining.instruction_tuning.utils import hashing


def load_dataset_from_source(
    dataset_name: str,
    dataset_split: str,
    dataset_revision: str,
) -> typing.Tuple[Dataset, typing.Optional[str]]:
    """
    Load dataset from local file or HuggingFace Hub.

    Args:
        dataset_name: Name or path of dataset
        dataset_split: Dataset split to load
        dataset_revision: Revision/branch of dataset

    Returns:
        Tuple of (dataset, commit_hash)
    """
    # Handle local files
    if os.path.exists(dataset_name) and dataset_name.endswith(".jsonl"):
        assert dataset_split == "train", "Only train split is supported for local jsonl files."
        dataset = load_dataset("json", data_files=dataset_name, split=dataset_split)
        return dataset, None

    elif os.path.exists(dataset_name) and dataset_name.endswith(".parquet"):
        assert dataset_split == "train", "Only train split is supported for local parquet files."
        dataset = load_dataset("parquet", data_files=dataset_name, split=dataset_split)
        return dataset, None

    else:
        # Load from HuggingFace Hub
        dataset_commit_hash = hashing.get_commit_hash(
            dataset_name, dataset_revision, "README.md", "dataset"
        )
        dataset = load_dataset(dataset_name, split=dataset_split, revision=dataset_revision)
        return dataset, dataset_commit_hash


def validate_dataset_splits(
    dataset_mixer_list: typing.List[str],
    dataset_mixer_list_splits: typing.List[str],
) -> typing.List[str]:
    """
    Validate and expand dataset splits to match dataset count.

    Args:
        dataset_mixer_list: List of datasets and weights
        dataset_mixer_list_splits: List of splits

    Returns:
        Expanded list of splits

    Raises:
        ValueError: If splits don't match dataset count
    """
    num_datasets = len(dataset_mixer_list) // 2

    if len(dataset_mixer_list_splits) == 1:
        print("Using the same split for all datasets")
        return [dataset_mixer_list_splits[0]] * num_datasets
    else:
        if len(dataset_mixer_list_splits) != num_datasets:
            raise ValueError(
                f"dataset_mixer_list_splits length must match number of datasets: "
                f"{len(dataset_mixer_list_splits)} != {num_datasets}"
            )
        return dataset_mixer_list_splits


def validate_mixer_list_format(dataset_mixer_list: typing.List[str]) -> None:
    """
    Validate that dataset mixer list has correct format.

    Args:
        dataset_mixer_list: List to validate

    Raises:
        AssertionError: If format is invalid
    """
    assert len(dataset_mixer_list) % 2 == 0, (
        f"Dataset mixer list length must be even: {dataset_mixer_list}"
    )
