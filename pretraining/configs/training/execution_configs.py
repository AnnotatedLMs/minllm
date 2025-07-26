# Standard Library
import abc
import enum
import typing

# Third Party
import pydantic
import torch

# Project
from pretraining.configs import base
from pretraining.utils.training import distributed


class ExecutionStrategy(enum.Enum):
    """Training execution strategy."""

    SINGLE = "single"
    """Train on a single device."""

    DDP = "ddp"
    """Distributed Data Parallel training."""


class DDPGradSyncMode(enum.Enum):
    """Gradient synchronization mode for DDP."""

    BATCH = "batch"
    """
    Synchronize gradients after computation at each bucket only at the last micro-batch.
    This is slightly faster than gradient syncs across each micro-batch but will consume more memory.
    Can use this mode only when `find_unused_params` is set to False.
    """

    MICRO_BATCH = "micro_batch"
    """
    Synchronize gradients after computation at each bucket per micro-batch.
    This will be slightly slower than gradient sync at the last micro-batch, but will consume less memory.
    Can use this mode with both option of `find_unused_params` but specifically recommended to use with `find_unused_params`
    set to True, to prevent errors.
    """


class BaseDeviceConfig(base.BaseConfig, abc.ABC):
    """Base class for device configuration with common device setup logic."""

    @abc.abstractmethod
    def setup_device(self) -> torch.device:
        """Setup and return the device for this strategy."""
        pass

    def _setup_cuda_device(self) -> torch.device:
        """Common CUDA device setup logic."""
        local_rank = distributed.get_local_rank()
        torch.cuda.set_device(f"cuda:{local_rank}")
        torch.cuda.empty_cache()
        return torch.device("cuda")

    def _setup_auto_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return self._setup_cuda_device()
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


class SingleConfig(BaseDeviceConfig):
    """Configuration for single-device training."""

    device: str = "auto"
    """Device specification: 'auto', 'cuda', 'mps', or 'cpu'."""

    def setup_device(self) -> torch.device:
        """Setup device for single-device training."""
        if self.device == "auto":
            return self._setup_auto_device()
        elif self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS not available.")
        elif self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA not available.")
        else:
            return torch.device(self.device)


class DDPConfig(BaseDeviceConfig):
    """Configuration for Distributed Data Parallel training."""

    backend: typing.Literal["nccl", "gloo"] = "nccl"
    """DDP backend to use."""

    find_unused_params: bool = False
    """
    (from torch documentation)

    This mode allows running backward on a subgraph of the model, and DDP finds out which parameters
    are involved in the backward pass by traversing the autograd graph from the model output and marking
    all unused parameters as ready for reduction. Note that traversing the autograd graph introduces extra overheads,
    so applications should only set find_unused_parameters to True when necessary.
    """

    grad_sync_mode: DDPGradSyncMode = DDPGradSyncMode.BATCH
    """
    Gradient sync mode for DDP.

    Note: When `find_unused_params` is set, set `grad_sync_mode` to `micro_batch` as different micro-batches might activate
    different parts of the model, ex- MOEs.
    """

    bucket_cap_mb: typing.Optional[int] = None
    """DDP gradient bucketing size in MB."""

    @pydantic.model_validator(mode="after")
    def validate_grad_sync_mode(self):
        if self.find_unused_params and self.grad_sync_mode != DDPGradSyncMode.MICRO_BATCH:
            raise ValueError(
                "When find_unused_params=True, grad_sync_mode must be MICRO_BATCH "
                "to avoid errors with different micro-batches activating different parameters"
            )
        return self

    def setup_device(self) -> torch.device:
        """Setup device for DDP - always use CUDA with local rank."""
        if not torch.cuda.is_available():
            raise ValueError("DDP requires CUDA")
        return self._setup_cuda_device()


class ExecutionConfig(base.BaseConfig):
    """Complete execution configuration."""

    strategy: ExecutionStrategy = ExecutionStrategy.SINGLE
    """Which execution strategy to use."""

    # Strategy-specific configs (only one should be set based on strategy)
    ddp: typing.Optional[DDPConfig] = None
    single: typing.Optional[SingleConfig] = None

    @pydantic.model_validator(mode="after")
    def ensure_strategy_config(self):
        # Ensure correct config is provided for strategy
        if self.strategy == ExecutionStrategy.DDP and self.ddp is None:
            self.ddp = DDPConfig()
        elif self.strategy == ExecutionStrategy.SINGLE and self.single is None:
            self.single = SingleConfig()
        return self

    def validate(self):
        """Validate configuration consistency."""
        if self.strategy == ExecutionStrategy.DDP and self.ddp is None:
            raise ValueError("DDP strategy requires ddp config")
        elif self.strategy == ExecutionStrategy.SINGLE and self.single is None:
            raise ValueError("Single strategy requires single device config")

    def setup_device(self) -> torch.device:
        """Setup device based on the execution strategy."""
        if self.strategy == ExecutionStrategy.DDP:
            return self.ddp.setup_device()
        elif self.strategy == ExecutionStrategy.SINGLE:
            return self.single.setup_device()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
