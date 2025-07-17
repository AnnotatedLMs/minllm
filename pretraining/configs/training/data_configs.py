# Standard Library
import dataclasses

# TODO: adjust based on dataset/ dataloader requirements


@dataclasses.dataclass
class DataConfig:
    """Configuration for data loading."""

    dataset: str
    data_dir: str
    num_workers: int  # Number of data loading workers
    pin_memory: bool  # Pin memory for faster GPU transfer
