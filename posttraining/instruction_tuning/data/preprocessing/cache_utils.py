# Standard Library
import hashlib
import json
import typing
from dataclasses import asdict

# Project
from posttraining.instruction_tuning.configs.data import dataset_config
from posttraining.instruction_tuning.configs.data import tokenizer_config


def compute_config_hash(
    dcs: typing.List[dataset_config.DatasetConfig], tc: tokenizer_config.TokenizerConfig
) -> str:
    """
    Compute a deterministic hash of configs for caching.

    Args:
        dcs: List of dataset configurations
        tc: Tokenizer configuration

    Returns:
        10-character hash string
    """
    # Convert configs to dictionaries, excluding None values
    dc_dicts = [{k: v for k, v in asdict(dc).items() if v is not None} for dc in dcs]
    tc_dict = {k: v for k, v in asdict(tc).items() if v is not None}

    # Combine into single dict
    combined_dict = {"dataset_configs": dc_dicts, "tokenizer_config": tc_dict}

    # Create hash
    config_str = json.dumps(combined_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:10]
