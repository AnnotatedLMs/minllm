# Standard Library
import typing
from dataclasses import dataclass
from dataclasses import field

# Project
from posttraining.instruction_tuning.data.utils import loading
from posttraining.instruction_tuning.data.utils import sampling


@dataclass
class DatasetConfig:
    """Configuration for a single dataset in the training mixture."""

    dataset_name: str
    dataset_split: str
    dataset_revision: str
    dataset_range: typing.Optional[int] = None
    transform_fn: typing.List[str] = field(default_factory=list)
    transform_fn_args: typing.List[typing.Dict[str, typing.Any]] = field(default_factory=list)
    target_columns: typing.Optional[typing.List[str]] = None

    # For tracking purposes
    dataset_commit_hash: typing.Optional[str] = None
    frac_or_num_samples: typing.Optional[typing.Union[int, float]] = None
    original_dataset_size: typing.Optional[int] = None
    is_upsampled: bool = False

    def __post_init__(self):
        """Load the dataset after initialization."""
        # Load dataset from source
        self.dataset, self.dataset_commit_hash = loading.load_dataset_from_source(
            dataset_name=self.dataset_name,
            dataset_split=self.dataset_split,
            dataset_revision=self.dataset_revision,
        )

        # Set default range if not specified
        if self.dataset_range is None:
            dataset_range = len(self.dataset)
            self.update_range(dataset_range)

    def update_range(self, dataset_range: int):
        """
        Update the dataset range and handle upsampling if needed.

        Args:
            dataset_range: Target number of samples
        """
        self.dataset_range = dataset_range
        original_size = len(self.dataset)
        self.original_dataset_size = original_size

        self.dataset = sampling.select_dataset_samples(
            dataset=self.dataset,
            target_size=self.dataset_range,
        )
        self.is_upsampled = dataset_range > original_size
