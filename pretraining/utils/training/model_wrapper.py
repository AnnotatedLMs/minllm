# Standard Library
import typing

# Third Party
import packaging.version
import torch
import torch.distributed.device_mesh
import torch.nn
import torch.nn.parallel
from torch.distributed import fsdp

# Project
from pretraining.configs.training import execution_configs
from pretraining.utils import torch_utils
from pretraining.utils.training import distributed


def wrap_model(
    model: torch.nn.Module,
    execution_config: execution_configs.ExecutionConfig,
    mixed_precision_dtype: typing.Optional[torch.dtype] = None,
) -> torch.nn.Module:
    """Wrap model based on execution strategy (DDP, FSDP, or Single).

    Args:
        model: The model to wrap
        execution_config: Execution configuration containing strategy and settings
        mixed_precision_dtype: Optional dtype for mixed precision training

    Returns:
        Wrapped model appropriate for the execution strategy
    """
    strategy = execution_config.strategy

    if strategy == execution_configs.ExecutionStrategy.DDP:
        return _wrap_ddp(model, execution_config.ddp)

    elif strategy == execution_configs.ExecutionStrategy.FSDP:
        return _wrap_fsdp(model, execution_config.fsdp, mixed_precision_dtype)

    else:  # SINGLE
        return distributed.SingleAccelerator(model)


def _wrap_ddp(
    model: torch.nn.Module,
    ddp_config: execution_configs.DDPConfig,
) -> torch.nn.parallel.DistributedDataParallel:
    """Wrap model with DistributedDataParallel."""
    device = torch.cuda.current_device() if torch.cuda.is_available() else None

    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device] if device is not None else None,
        find_unused_parameters=ddp_config.find_unused_params,
        bucket_cap_mb=ddp_config.bucket_cap_mb,
    )


def _wrap_fsdp(
    model: torch.nn.Module,
    fsdp_config: execution_configs.FSDPConfig,
    mixed_precision_dtype: typing.Optional[torch.dtype] = None,
) -> fsdp.FullyShardedDataParallel:
    """Wrap model with FullyShardedDataParallel."""

    # Get wrap policy from model
    if not hasattr(model, "get_fsdp_wrap_policy"):
        raise AttributeError(
            f"Model {type(model).__name__} must implement get_fsdp_wrap_policy() method for FSDP"
        )
    wrap_policy = model.get_fsdp_wrap_policy(fsdp_config.wrapping_strategy)

    # Setup mixed precision if configured
    mixed_precision = None
    if mixed_precision_dtype is not None and fsdp_config.precision is not None:
        if fsdp_config.precision == execution_configs.FSDPPrecision.PURE:
            mixed_precision = fsdp.MixedPrecision(
                param_dtype=mixed_precision_dtype,
                reduce_dtype=mixed_precision_dtype,
                buffer_dtype=mixed_precision_dtype,
            )
        elif fsdp_config.precision == execution_configs.FSDPPrecision.MIXED:
            mixed_precision = fsdp.MixedPrecision(
                param_dtype=mixed_precision_dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=mixed_precision_dtype,
            )

    # Setup device mesh for hybrid sharding if needed
    device_mesh = None
    hybrid_sharding_kwargs = {}
    if fsdp_config.sharding_strategy in [
        execution_configs.FSDPShardingStrategy.HYBRID_SHARD,
        execution_configs.FSDPShardingStrategy._HYBRID_SHARD_ZERO2,
    ]:
        device_mesh = _setup_hybrid_sharding_device_mesh(fsdp_config)
        hybrid_sharding_kwargs["device_mesh"] = device_mesh

    # Map config sharding strategy to PyTorch enum
    sharding_strategy = _get_sharding_strategy(fsdp_config.sharding_strategy)

    # Setup parameter initialization function for newer PyTorch versions
    param_init_fn = _get_param_init_fn()

    # Create FSDP wrapped model
    wrapped_model = fsdp.FullyShardedDataParallel(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        auto_wrap_policy=wrap_policy,
        use_orig_params=fsdp_config.use_orig_params,
        limit_all_gathers=fsdp_config.limit_all_gathers,
        device_id=distributed.get_local_rank(),
        param_init_fn=param_init_fn,
        backward_prefetch=fsdp.BackwardPrefetch.BACKWARD_PRE
        if fsdp_config.backward_prefetch
        else None,
        forward_prefetch=fsdp_config.forward_prefetch,
        cpu_offload=fsdp.CPUOffload(offload_params=fsdp_config.cpu_offload),
        **hybrid_sharding_kwargs,
    )

    # Initialize parameters if needed
    if param_init_fn is not None:
        model.reset_parameters()

    return wrapped_model


def _setup_hybrid_sharding_device_mesh(
    fsdp_config: execution_configs.FSDPConfig,
) -> torch.distributed.device_mesh.DeviceMesh:
    """Setup device mesh for hybrid sharding."""
    num_model_replicas = fsdp_config.hybrid_sharding_num_model_replicas or (
        distributed.get_world_size() // distributed.get_local_world_size()
    )

    if distributed.get_world_size() % num_model_replicas != 0:
        raise ValueError(
            f"hybrid_sharding_num_model_replicas ({num_model_replicas}) "
            f"must divide world size ({distributed.get_world_size()})"
        )

    return torch.distributed.device_mesh.init_device_mesh(
        "cuda", (num_model_replicas, distributed.get_world_size() // num_model_replicas)
    )


def _get_sharding_strategy(
    config_strategy: execution_configs.FSDPShardingStrategy,
) -> fsdp.ShardingStrategy:
    """Map config sharding strategy enum to PyTorch FSDP enum."""
    strategy_map = {
        execution_configs.FSDPShardingStrategy.FULL_SHARD: fsdp.ShardingStrategy.FULL_SHARD,
        execution_configs.FSDPShardingStrategy.SHARD_GRAD_OP: fsdp.ShardingStrategy.SHARD_GRAD_OP,
        execution_configs.FSDPShardingStrategy.NO_SHARD: fsdp.ShardingStrategy.NO_SHARD,
        execution_configs.FSDPShardingStrategy.HYBRID_SHARD: fsdp.ShardingStrategy.HYBRID_SHARD,
        execution_configs.FSDPShardingStrategy._HYBRID_SHARD_ZERO2: fsdp.ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    return strategy_map[config_strategy]


def _get_param_init_fn() -> typing.Optional[typing.Callable]:
    """Get parameter initialization function for FSDP based on PyTorch version."""
    if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.1.0"):
        # This prevents any parameters from being initialized twice
        def dummy_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=torch_utils.get_default_device())

        return dummy_init_fn
    return None
