# Standard Library
import typing

# Third Party
import numpy as np
from datasets import Dataset


def select_dataset_samples(
    dataset: Dataset,
    target_size: int,
    random_seed: int = 42,
) -> Dataset:
    """
    Select or upsample dataset to target size.

    If target_size > dataset size, repeat samples to reach target.

    Args:
        dataset: Source dataset
        target_size: Number of samples to select
        random_seed: Random seed for reproducibility

    Returns:
        Dataset with target_size samples
    """
    original_size = len(dataset)

    # Calculate how many full repeats and extra samples
    full_repeats = target_size // original_size
    extra_samples = target_size % original_size

    # Create indices for sampling
    indices = []

    # Add full repeats
    for _ in range(full_repeats):
        indices.extend(range(original_size))

    # Add randomly sampled extra samples
    if extra_samples > 0:
        # Use numpy for reproducible random sampling
        rng = np.random.RandomState(random_seed)
        extra_indices = rng.choice(original_size, size=extra_samples, replace=False)
        indices.extend(extra_indices.tolist())

    if target_size > original_size:
        print(
            f"Upsampling dataset from {original_size} to {target_size} samples "
            f"({full_repeats} full repeats + {extra_samples} random samples)"
        )

    return dataset.select(indices)


def calculate_dataset_range(
    frac_or_num_samples: typing.Union[int, float],
    original_size: int,
) -> int:
    """
    Calculate target dataset size from fractional or absolute specification.

    Args:
        frac_or_num_samples: Either fraction of dataset (float) or absolute count (int)
        original_size: Size of original dataset

    Returns:
        Target number of samples
    """
    if isinstance(frac_or_num_samples, int) and frac_or_num_samples > original_size:
        # Absolute number larger than dataset - use for upsampling
        return frac_or_num_samples
    elif isinstance(frac_or_num_samples, float):
        # Fractional sampling (can be > 1.0 for upsampling)
        return int(frac_or_num_samples * original_size)
    else:
        # Integer <= dataset size
        return int(frac_or_num_samples)


def parse_weight_specification(weight_str: str) -> typing.Union[int, float]:
    """
    Parse weight string as float or int.

    Args:
        weight_str: Weight specification as string

    Returns:
        Weight as float or int
    """
    if "." in weight_str:
        return float(weight_str)
    else:
        return int(weight_str)
