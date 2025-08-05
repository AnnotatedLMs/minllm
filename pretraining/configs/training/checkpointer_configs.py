# Standard Library
import enum
import typing

# Third Party
import pydantic

# Project
from pretraining.configs import base


class CheckpointType(enum.Enum):
    """Types of checkpoints."""

    SHARDED = "sharded"
    """Distributed checkpoint where each rank saves its own shard."""

    UNSHARDED = "unsharded"
    """Full model checkpoint gathered on rank 0."""

    SHARDED_EPHEMERAL = "sharded_ephemeral"
    """Temporary sharded checkpoint, only most recent is kept."""


class ShardedCheckpointerType(enum.Enum):
    """Sharded checkpointer implementations."""

    TORCH_NEW = "torch_new"
    """PyTorch's new distributed checkpoint format."""

    TORCH_LEGACY = "torch_legacy"
    """PyTorch's legacy distributed checkpoint format."""

    LOCAL = "local"
    """Custom local checkpointing."""

    OLMO_CORE = "olmo_core"
    """OLMo core checkpointing (if available)."""


class CheckpointerConfig(base.BaseConfig):
    """Configuration for model checkpointing."""

    save_dir: str
    save_interval: int = pydantic.Field(gt=0, description="Steps between regular checkpoints")
    save_best: bool
    keep_last_n: int = pydantic.Field(ge=0)
    resume_from: typing.Optional[str] = None  # Path to checkpoint to resume from

    # FSDP-specific options
    save_interval_unsharded: typing.Optional[int] = pydantic.Field(
        None, gt=0, description="Steps between unsharded checkpoints (expensive for large models)"
    )
    save_interval_ephemeral: typing.Optional[int] = pydantic.Field(
        None, gt=0, description="Steps between ephemeral checkpoints (only latest kept)"
    )
    keep_last_n_unsharded: int = pydantic.Field(
        -1, ge=-1, description="Number of unsharded checkpoints to keep (-1 = keep all)"
    )
    sharded_checkpointer: ShardedCheckpointerType = ShardedCheckpointerType.TORCH_LEGACY
    """Which sharded checkpointer implementation to use."""

    save_overwrite: bool = False
    """Whether to overwrite existing checkpoints."""

    remote_save_folder: typing.Optional[str] = None
    """Optional cloud bucket folder to upload checkpoints to."""
