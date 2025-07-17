# Standard Library
import dataclasses
import typing


@dataclasses.dataclass
class CheckpointerConfig:
    """Configuration for model checkpointing."""

    save_dir: str
    save_interval: int
    save_best: bool
    keep_last_n: int
    resume_from: typing.Optional[str] = None  # Path to checkpoint to resume from
