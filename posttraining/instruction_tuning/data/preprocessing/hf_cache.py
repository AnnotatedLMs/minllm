# Standard Library
import json
import typing
from dataclasses import asdict

# Third Party
from datasets import Dataset
from datasets import load_dataset
from huggingface_hub import ModelCard
from huggingface_hub import revision_exists

# Project
from posttraining.instruction_tuning.configs.data import dataset_config
from posttraining.instruction_tuning.configs.data import tokenizer_config
from posttraining.instruction_tuning.utils import auth


class DatasetTransformationCache:
    """HuggingFace Hub cache for transformed datasets."""

    # Cached datasets always use train split
    DEFAULT_SPLIT_FOR_CACHED_DATASET = "train"

    def __init__(self, config_hash: str, hf_entity: typing.Optional[str] = None):
        """
        Initialize HF Hub cache.

        Args:
            config_hash: Hash of the configuration
            hf_entity: HuggingFace entity name
        """
        self.config_hash = config_hash
        self.hf_entity = hf_entity or auth.hf_whoami()["name"]
        self.repo_name = f"{self.hf_entity}/dataset-mix-cached"

    def get_cache_url(self) -> str:
        """Get the URL for the cached dataset."""
        return f"https://huggingface.co/datasets/{self.repo_name}/tree/{self.config_hash}"

    def cache_exists(self) -> bool:
        """Check if cache exists on HF Hub."""
        return revision_exists(self.repo_name, self.config_hash, repo_type="dataset")

    def load_dataset(self) -> Dataset:
        """
        Load dataset from HF Hub cache.

        Returns:
            Cached dataset
        """
        print(f"✅ Found cached dataset at {self.get_cache_url()}")
        return load_dataset(
            self.repo_name, split=self.DEFAULT_SPLIT_FOR_CACHED_DATASET, revision=self.config_hash
        )

    def create_model_card(
        self, dcs: typing.List[dataset_config.DatasetConfig], tc: tokenizer_config.TokenizerConfig
    ) -> ModelCard:
        """
        Create model card for cached dataset.

        Args:
            dcs: Dataset configurations
            tc: Tokenizer configuration

        Returns:
            Model card for the dataset
        """
        return ModelCard(
            f"""\
---
tags: [open-instruct]
---

# Cached Tokenized Datasets

## Summary

This is a cached dataset produced by the instruction tuning pipeline.

## Configuration

`tokenizer_config.TokenizerConfig`:
```json
{json.dumps(asdict(tc), indent=2)}
```

`List[dataset_config.DatasetConfig]`:
```json
{json.dumps([asdict(dc) for dc in dcs], indent=2)}
```
"""
        )

    def save_dataset(
        self,
        dataset: Dataset,
        dcs: typing.List[dataset_config.DatasetConfig],
        tc: tokenizer_config.TokenizerConfig,
    ) -> Dataset:
        """
        Save dataset to HF Hub cache.

        Args:
            dataset: Dataset to save
            dcs: Dataset configurations
            tc: Tokenizer configuration

        Returns:
            Loaded dataset from HF Hub
        """
        # Push dataset to hub
        dataset.push_to_hub(
            self.repo_name,
            private=True,
            revision=self.config_hash,
            commit_message=f"Cache combined dataset with configs hash: {self.config_hash}",
        )
        print(f"🚀 Pushed transformed dataset to {self.get_cache_url()}")

        # Create and push model card
        model_card = self.create_model_card(dcs, tc)
        model_card.push_to_hub(self.repo_name, repo_type="dataset", revision=self.config_hash)

        # Load again to ensure it's downloaded to HF cache
        return self.load_dataset()

    def load_or_transform_dataset(
        self,
        dcs: typing.List[dataset_config.DatasetConfig],
        tc: tokenizer_config.TokenizerConfig,
        transform_fn: typing.Callable,
        dataset_skip_cache: bool = False,
    ) -> Dataset:
        """
        Load dataset from HF Hub cache or transform and cache it.

        Args:
            dcs: Dataset configurations
            tc: Tokenizer configuration
            transform_fn: Function to transform datasets
            dataset_skip_cache: Whether to skip cache

        Returns:
            Transformed dataset
        """
        # Check if revision exists
        if self.cache_exists() and not dataset_skip_cache:
            return self.load_dataset()

        print("Cache not found, transforming datasets...")

        # Transform datasets
        combined_dataset, _ = transform_fn(dcs, tc)

        if dataset_skip_cache:
            return combined_dataset

        # Save to HF Hub
        return self.save_dataset(combined_dataset, dcs, tc)
