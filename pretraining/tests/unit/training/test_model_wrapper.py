"""Unit tests for model wrapper functionality."""

# Standard Library
import unittest

# Third Party
import torch
import torch.nn as nn

# Project
from pretraining.configs.training import execution_configs
from pretraining.utils.training import dist_utils
from pretraining.utils.training import model_wrapper


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1000)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        return self.linear2(x)

    def get_fsdp_wrappable_modules(self):
        """Return wrappable modules for FSDP."""
        return {nn.Linear}

    def get_fsdp_wrap_policy(self, strategy):
        """Return wrap policy for FSDP."""
        if strategy == execution_configs.FSDPWrapStrategy.BY_BLOCK:

            def policy(module, recurse, nonwrapped_numel):
                return isinstance(module, nn.Linear)

            return policy
        return None


class TestModelWrapper(unittest.TestCase):
    """Test model wrapper functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.model = DummyModel()
        self.execution_config = execution_configs.ExecutionConfig(
            strategy=execution_configs.ExecutionStrategy.SINGLE
        )

    def test_single_device_wrapper(self):
        """Test single device wrapper."""
        wrapped = model_wrapper.wrap_model(self.model, self.execution_config, torch.float32)

        # Should return SingleAccelerator
        self.assertIsInstance(wrapped, dist_utils.SingleAccelerator)
        self.assertEqual(wrapped.module, self.model)

    def test_mixed_precision_config(self):
        """Test mixed precision configuration."""
        # Test bfloat16
        wrapped = model_wrapper.wrap_model(self.model, self.execution_config, torch.bfloat16)

        # Model should still be wrapped
        self.assertIsInstance(wrapped, dist_utils.SingleAccelerator)

        # Test float16
        wrapped = model_wrapper.wrap_model(self.model, self.execution_config, torch.float16)

        self.assertIsInstance(wrapped, dist_utils.SingleAccelerator)


if __name__ == "__main__":
    unittest.main()
