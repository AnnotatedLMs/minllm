# Standard Library
import abc
import enum
import typing

# Third Party
import pydantic
import torch

# Project
from pretraining.configs import base
from pretraining.utils.training import dist_utils


class ExecutionStrategy(enum.Enum):
    """Training execution strategy."""

    SINGLE = "single"
    """Train on a single device."""

    DDP = "ddp"
    """Distributed Data Parallel training."""

    FSDP = "fsdp"
    """Fully Sharded Data Parallel training."""


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


class FSDPShardingStrategy(enum.Enum):
    """FSDP sharding strategies."""

    FULL_SHARD = "full_shard"
    """Shard model parameters, gradients and optimizer states across all ranks."""

    SHARD_GRAD_OP = "shard_grad_op"
    """Shard gradients and optimizer states only (keep parameters replicated)."""

    NO_SHARD = "no_shard"
    """No sharding - similar to DDP."""

    HYBRID_SHARD = "hybrid_shard"
    """Hybrid sharding across nodes."""

    _HYBRID_SHARD_ZERO2 = "_hybrid_shard_zero2"
    """Hybrid sharding with Zero2 strategy."""


class FSDPWrapStrategy(enum.Enum):
    """FSDP model wrapping strategies."""

    BY_BLOCK = "by_block"
    """Wrap each transformer block with its own FSDP instance."""

    BY_BLOCK_AND_SIZE = "by_block_and_size"
    """Like 'by_block' but embeddings and output layers wrapped separately."""

    BY_BLOCK_GROUP = "by_block_group"
    """Wrap block groups together (requires block_group_size > 1)."""

    BY_BLOCK_GROUP_AND_SIZE = "by_block_group_and_size"
    """Like 'by_block_group' but embeddings and output layers wrapped separately."""

    SIZE_BASED = "size_based"
    """Use PyTorch's default size-based auto wrap policy."""

    ONE_IN_TWO = "one_in_two"
    """Wrap every other layer."""

    ONE_IN_THREE = "one_in_three"
    """Wrap every third layer."""

    ONE_IN_FOUR = "one_in_four"
    """Wrap every fourth layer."""

    ONE_IN_FIVE = "one_in_five"
    """Wrap every fifth layer."""


class ActivationCheckpointingStrategy(enum.Enum):
    """Activation checkpointing strategies for memory optimization."""

    NONE = "none"
    """No activation checkpointing."""

    WHOLE_LAYER = "whole_layer"
    """Checkpoint every transformer layer."""

    ONE_IN_TWO = "one_in_two"
    """Checkpoint one in two transformer layers."""

    ONE_IN_THREE = "one_in_three"
    """Checkpoint one in three transformer layers."""

    ONE_IN_FOUR = "one_in_four"
    """Checkpoint one in four transformer layers."""

    ONE_IN_EIGHT = "one_in_eight"
    """Checkpoint one in eight transformer layers."""

    TWO_IN_THREE = "two_in_three"
    """Checkpoint two out of every three transformer layers."""

    THREE_IN_FOUR = "three_in_four"
    """Checkpoint three out of four transformer layers."""

    FINE_GRAINED = "fine_grained"
    """Focus checkpointing on where it is cheap to recompute and saves most memory."""


class FSDPPrecision(enum.Enum):
    """FSDP mixed precision modes."""

    PURE = "pure"
    """All operations in autocast precision (bf16/fp16)."""

    MIXED = "mixed"
    """Parameters and buffers in autocast precision, reductions in fp32."""


class BaseDeviceConfig(base.BaseConfig, abc.ABC):
    """Base class for device configuration with common device setup logic."""

    @abc.abstractmethod
    def setup_device(self) -> torch.device:
        """Setup and return the device for this strategy."""
        pass

    def _setup_cuda_device(self) -> torch.device:
        """Common CUDA device setup logic."""
        local_rank = dist_utils.get_local_rank()
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


class FSDPConfig(BaseDeviceConfig):
    """Configuration for Fully Sharded Data Parallel training."""

    sharding_strategy: FSDPShardingStrategy = FSDPShardingStrategy.FULL_SHARD
    """How to shard model parameters, gradients, and optimizer states."""

    wrapping_strategy: typing.Optional[FSDPWrapStrategy] = None
    """How to wrap model layers. If None, wrap entire model with single FSDP instance."""

    precision: typing.Optional[FSDPPrecision] = FSDPPrecision.PURE
    """Mixed precision configuration for FSDP."""

    use_orig_params: bool = True
    """Must be True for torch.compile() or parameter norm tracking."""

    cpu_offload: bool = False
    """Whether to offload parameters to CPU when not in use."""

    backward_prefetch: bool = True
    """Whether to prefetch next layer's parameters during backward pass."""

    forward_prefetch: bool = False
    """Whether to prefetch next layer's parameters during forward pass."""

    limit_all_gathers: bool = True
    """Limit the number of concurrent all-gather operations for memory efficiency."""

    hybrid_sharding_num_model_replicas: typing.Optional[int] = None
    """
    Number of model replicas for hybrid sharding. If None, defaults to
    world_size // local_world_size (one replica per node).
    """

    min_params_size: int = int(1e8)
    """Minimum parameter size for SIZE_BASED wrapping strategy."""

    # Block grouping for wrapping strategies
    block_group_size: int = pydantic.Field(default=1, ge=1)
    """Number of transformer blocks to group together for FSDP wrapping."""

    # Activation checkpointing
    activation_checkpointing: typing.Optional[ActivationCheckpointingStrategy] = None
    """Strategy for activation checkpointing to save memory."""

    activation_checkpointing_reentrant: bool = False
    """Whether to use reentrant activation checkpointing (slower but more compatible)."""

    # Synchronization
    sync_module_states: bool = True
    """Whether to synchronize module buffers across ranks during initialization."""

    # State dict configuration for checkpointing
    state_dict_type: typing.Literal["full", "local", "sharded"] = "full"
    """Type of state dict to use for checkpointing."""

    state_dict_rank0_only: bool = True
    """Whether only rank 0 should save unsharded checkpoints."""

    # Advanced options
    ignored_modules: typing.Optional[typing.List[str]] = None
    """List of module names to ignore when wrapping with FSDP."""

    backward_prefetch_policy: typing.Literal["backward_pre", "backward_post", "none"] = (
        "backward_pre"
    )
    """When to prefetch next layer's parameters during backward pass."""

    @pydantic.model_validator(mode="after")
    def validate_block_group_wrapping(self):
        """Validate block group wrapping configuration."""
        if self.wrapping_strategy in [
            FSDPWrapStrategy.BY_BLOCK_GROUP,
            FSDPWrapStrategy.BY_BLOCK_GROUP_AND_SIZE,
        ]:
            if self.block_group_size <= 1:
                raise ValueError(
                    f"'{self.wrapping_strategy.value}' wrapping strategy requires "
                    f"block_group_size > 1, got {self.block_group_size}"
                )
        return self

    def setup_device(self) -> torch.device:
        """Setup device for FSDP - always use CUDA."""
        if not torch.cuda.is_available():
            raise ValueError("FSDP requires CUDA")
        return self._setup_cuda_device()


class ExecutionConfig(base.BaseConfig):
    """Complete execution configuration."""

    strategy: ExecutionStrategy = ExecutionStrategy.SINGLE
    """Which execution strategy to use."""

    # Strategy-specific configs (only one should be set based on strategy)
    ddp: typing.Optional[DDPConfig] = None
    fsdp: typing.Optional[FSDPConfig] = None
    single: typing.Optional[SingleConfig] = None

    @pydantic.model_validator(mode="after")
    def ensure_strategy_config(self):
        # Ensure correct config is provided for strategy
        if self.strategy == ExecutionStrategy.DDP and self.ddp is None:
            self.ddp = DDPConfig()
        elif self.strategy == ExecutionStrategy.FSDP and self.fsdp is None:
            self.fsdp = FSDPConfig()
        elif self.strategy == ExecutionStrategy.SINGLE and self.single is None:
            self.single = SingleConfig()
        return self

    def validate(self):
        """Validate configuration consistency."""
        if self.strategy == ExecutionStrategy.DDP and self.ddp is None:
            raise ValueError("DDP strategy requires ddp config")
        elif self.strategy == ExecutionStrategy.FSDP and self.fsdp is None:
            raise ValueError("FSDP strategy requires fsdp config")
        elif self.strategy == ExecutionStrategy.SINGLE and self.single is None:
            raise ValueError("Single strategy requires single device config")

    def setup_device(self) -> torch.device:
        """Setup device based on the execution strategy."""
        if self.strategy == ExecutionStrategy.DDP:
            return self.ddp.setup_device()
        elif self.strategy == ExecutionStrategy.FSDP:
            return self.fsdp.setup_device()
        elif self.strategy == ExecutionStrategy.SINGLE:
            return self.single.setup_device()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
