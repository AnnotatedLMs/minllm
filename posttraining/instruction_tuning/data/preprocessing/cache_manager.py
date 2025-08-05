# Re-export cache classes and utilities for backward compatibility

# Project
from posttraining.instruction_tuning.data.preprocessing import cache_utils
from posttraining.instruction_tuning.data.preprocessing import hf_cache
from posttraining.instruction_tuning.data.preprocessing import local_cache

compute_config_hash = cache_utils.compute_config_hash
DatasetTransformationCache = hf_cache.DatasetTransformationCache
LocalDatasetTransformationCache = local_cache.LocalDatasetTransformationCache

__all__ = [
    "compute_config_hash",
    "DatasetTransformationCache",
    "LocalDatasetTransformationCache",
]
