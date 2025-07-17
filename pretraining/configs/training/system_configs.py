# Standard Library
import dataclasses
import typing


@dataclasses.dataclass
class DeviceConfig:
    """Configuration for compute device and precision."""

    device: str  # e.g. 'cuda', 'cuda:0', 'cpu'
    dtype: typing.Literal["float32", "float16", "bfloat16"]

    @property
    def device_type(self) -> str:
        """Extract device type for autocast."""
        # Handle cases like 'cuda:0' -> 'cuda'
        return self.device.split(":")[0]


@dataclasses.dataclass
class TorchCompilationConfig:
    """Configuration for model compilation/optimization."""

    compile: bool  # PyTorch 2.0 compile
    compile_mode: typing.Optional[typing.Literal["default", "reduce-overhead", "max-autotune"]] = (
        None
    )


@dataclasses.dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    backend: typing.Literal["nccl", "gloo"]  # DDP backend
    find_unused_parameters: bool  # Whether to find unused params in DDP
    ddp_bucket_cap_mb: typing.Optional[int] = None  # DDP gradient bucketing size
