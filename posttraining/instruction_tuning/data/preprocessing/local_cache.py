# Standard Library
import json
import os
import typing
from dataclasses import asdict

# Third Party
from datasets import Dataset

# Project
from posttraining.instruction_tuning.configs.data import dataset_config
from posttraining.instruction_tuning.configs.data import tokenizer_config


class LocalDatasetTransformationCache:
    """Local filesystem cache for transformed datasets."""

    def __init__(self, config_hash: str, dataset_local_cache_dir: str):
        """
        Initialize local cache.

        Args:
            config_hash: Hash of the configuration
            dataset_local_cache_dir: Directory for cache storage
        """
        self.config_hash = config_hash
        self.dataset_local_cache_dir = dataset_local_cache_dir
        os.makedirs(dataset_local_cache_dir, exist_ok=True)

    def get_cache_path(self) -> str:
        """Get the path to the cached dataset."""
        return os.path.join(self.dataset_local_cache_dir, self.config_hash)

    def save_config(
        self,
        config_hash: str,
        dcs: typing.List[dataset_config.DatasetConfig],
        tc: tokenizer_config.TokenizerConfig,
    ):
        """
        Save configuration to JSON file.

        Args:
            config_hash: Configuration hash
            dcs: Dataset configurations
            tc: Tokenizer configuration
        """
        config_path = os.path.join(self.get_cache_path(), "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        config_dict = {
            "tokenizer_config": asdict(tc),
            "dataset_configs": [asdict(dc) for dc in dcs],
            "config_hash": config_hash,
        }

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def save_statistics(self, statistics: typing.Dict[str, typing.Any]) -> None:
        """
        Save dataset statistics to cache.

        Args:
            statistics: Dataset statistics dictionary
        """
        stats_path = os.path.join(self.get_cache_path(), "dataset_statistics.json")
        with open(stats_path, "w") as f:
            json.dump(statistics, f, indent=2)

    def load_statistics(self) -> typing.Dict[str, typing.Any]:
        """
        Load dataset statistics from cache.

        Returns:
            Statistics dictionary or default if not found
        """
        stats_path = os.path.join(self.get_cache_path(), "dataset_statistics.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                return json.load(f)
        return {"per_dataset_stats": [], "dataset_order": []}

    def cache_exists(self) -> bool:
        """Check if cache exists for this configuration."""
        return os.path.exists(self.get_cache_path())

    def load_dataset(self) -> Dataset:
        """
        Load dataset from cache.

        Returns:
            Cached dataset
        """
        print(f"✅ Found cached dataset at {self.get_cache_path()}")
        return Dataset.load_from_disk(self.get_cache_path(), keep_in_memory=True)

    def save_dataset(
        self,
        dataset: Dataset,
        dcs: typing.List[DatasetConfig],
        tc: TokenizerConfig,
        statistics: typing.Dict[str, typing.Any],
    ) -> Dataset:
        """
        Save dataset to cache.

        Args:
            dataset: Dataset to save
            dcs: Dataset configurations
            tc: Tokenizer configuration
            statistics: Dataset statistics

        Returns:
            Loaded dataset from cache
        """
        cache_path = self.get_cache_path()

        # Save dataset
        dataset.save_to_disk(cache_path)

        # Save configuration
        self.save_config(self.config_hash, dcs, tc)

        # Save statistics
        self.save_statistics(statistics)

        print(f"🚀 Saved transformed dataset to {cache_path}")

        # Load from disk to ensure it's in HF cache format
        return Dataset.load_from_disk(cache_path, keep_in_memory=True)

    def load_or_transform_dataset(
        self,
        dcs: typing.List[DatasetConfig],
        tc: TokenizerConfig,
        transform_fn: typing.Callable,
        dataset_skip_cache: bool = False,
    ) -> typing.Tuple[Dataset, typing.Dict[str, typing.Any]]:
        """
        Load dataset from cache or transform and cache it.

        Args:
            dcs: Dataset configurations
            tc: Tokenizer configuration
            transform_fn: Function to transform datasets
            dataset_skip_cache: Whether to skip cache

        Returns:
            Tuple of (dataset, statistics)
        """
        # Check if cache exists
        if self.cache_exists() and not dataset_skip_cache:
            dataset = self.load_dataset()
            statistics = self.load_statistics()
            return dataset, statistics

        print("Cache not found or skipped, transforming datasets...")

        # Transform datasets
        dataset, statistics = transform_fn(dcs, tc)

        if dataset_skip_cache:
            return dataset, statistics

        # Save to cache
        loaded_dataset = self.save_dataset(dataset, dcs, tc, statistics)
        return loaded_dataset, statistics
