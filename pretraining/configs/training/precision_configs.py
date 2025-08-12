# Standard Library
import enum
import typing

# Third Party
import torch

# Project
from pretraining.configs import base


class PrecisionType(str, enum.Enum):
    """Training precision types - following OLMo's approach."""

    # Full precision
    FP32 = "fp32"

    # Mixed precision with automatic mixed precision (AMP)
    AMP_BF16 = "amp_bf16"
    AMP_FP16 = "amp_fp16"


class PrecisionConfig(base.BaseConfig):
    """Configuration for training precision."""

    precision: PrecisionType = PrecisionType.FP32

    @property
    def is_mixed_precision(self) -> bool:
        """Check if using automatic mixed precision (AMP)."""
        return self.precision in [PrecisionType.AMP_BF16, PrecisionType.AMP_FP16]

    @property
    def get_dtype(self) -> typing.Optional[torch.dtype]:
        """Get the torch dtype for this precision config.

        Returns None for fp32 (no casting needed).
        """
        dtype_map = {
            PrecisionType.FP32: None,
            PrecisionType.AMP_BF16: torch.bfloat16,
            PrecisionType.AMP_FP16: torch.float16,
        }
        return dtype_map[self.precision]

    @property
    def autocast_dtype(self) -> typing.Optional[torch.dtype]:
        """Get the dtype for autocast context, or None if not using AMP."""
        if self.precision == PrecisionType.AMP_BF16:
            return torch.bfloat16
        elif self.precision == PrecisionType.AMP_FP16:
            return torch.float16
        return None
