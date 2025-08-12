# Standard Library
import pathlib
import unittest

# Third Party
import torch

# Project
from pretraining.common.patterns.architectures import deepseek3
from pretraining.common.patterns.architectures import gpt2
from pretraining.common.patterns.architectures import llama3
from pretraining.configs import loader
from pretraining.configs.model.architectures import deepseek
from pretraining.configs.model.architectures import gpt
from pretraining.configs.model.architectures import llama
from pretraining.configs.training import execution_configs


class TestModelStrategyMatrix(unittest.TestCase):
    """Test all combinations of models and execution strategies."""

    def setUp(self):
        """Set up test paths."""
        self.config_base_path = (
            pathlib.Path(__file__).parent.parent.parent / "configs" / "examples" / "debug"
        )

        # Define model configurations
        self.model_configs = [
            ("gpt2", gpt.GPT2Config, gpt2.GPT2),
            ("llama3", llama.Llama3Config, llama3.Llama3),
            ("deepseek3", deepseek.DeepSeek3Config, deepseek3.DeepSeek3),
        ]

        # Define execution strategies
        self.strategies = ["cpu", "single_gpu", "ddp", "fsdp"]

    def _get_config_path(self, model_name: str, strategy: str) -> pathlib.Path:
        """Get the path to a specific config file."""
        return self.config_base_path / model_name / f"{model_name}_debug_{strategy}.yaml"

    def _load_and_test_config(
        self, model_name: str, model_config_class, model_class, strategy: str
    ):
        """Load a config and test model instantiation."""
        config_path = self._get_config_path(model_name, strategy)

        # Skip if config doesn't exist
        if not config_path.exists():
            self.skipTest(f"Config not found: {config_path}")

        # Load the config
        trainer_config = loader.load_training_config(config_path, model_config_class)

        # Verify config loaded correctly
        self.assertIsNotNone(trainer_config)
        self.assertIsNotNone(trainer_config.llm)
        self.assertIsNotNone(trainer_config.training)

        # Check execution strategy matches
        self.assertEqual(
            trainer_config.training.execution.strategy,
            {"cpu": "single", "single_gpu": "single", "ddp": "ddp", "fsdp": "fsdp"}[strategy],
        )

        # Create model
        model = model_class.from_config(trainer_config.llm)
        self.assertIsNotNone(model)

        # Test basic forward pass
        batch_size = 2
        seq_length = 128
        vocab_size = trainer_config.llm.vocab_size

        # Create dummy input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

        # Device handling based on strategy
        if strategy == "cpu":
            device = torch.device("cpu")
        elif strategy in ["single_gpu", "ddp", "fsdp"]:
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")
            device = torch.device("cuda:0")

        # Move model and inputs to device
        model = model.to(device)
        input_ids = input_ids.to(device)

        # Test forward pass
        with torch.no_grad():
            output = model(input_ids)
            self.assertIsNotNone(output)

            # Check output shape
            if hasattr(output, "logits"):
                logits = output.logits
            else:
                logits = output

            expected_shape = (batch_size, seq_length, vocab_size)
            self.assertEqual(logits.shape, expected_shape)

        return model, trainer_config

    # CPU Tests
    def test_gpt2_cpu(self):
        """Test GPT-2 with CPU execution."""
        self._load_and_test_config("gpt2", gpt.GPT2Config, gpt2.GPT2, "cpu")

    def test_llama3_cpu(self):
        """Test Llama3 with CPU execution."""
        self._load_and_test_config("llama3", llama.Llama3Config, llama3.Llama3, "cpu")

    def test_deepseek3_cpu(self):
        """Test DeepSeek3 with CPU execution."""
        self._load_and_test_config(
            "deepseek3", deepseek.DeepSeek3Config, deepseek3.DeepSeek3, "cpu"
        )

    # Single GPU Tests
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpt2_single_gpu(self):
        """Test GPT-2 with single GPU execution."""
        self._load_and_test_config("gpt2", gpt.GPT2Config, gpt2.GPT2, "single_gpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_llama3_single_gpu(self):
        """Test Llama3 with single GPU execution."""
        self._load_and_test_config("llama3", llama.Llama3Config, llama3.Llama3, "single_gpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_deepseek3_single_gpu(self):
        """Test DeepSeek3 with single GPU execution."""
        self._load_and_test_config(
            "deepseek3", deepseek.DeepSeek3Config, deepseek3.DeepSeek3, "single_gpu"
        )


class TestWrappingStrategies(unittest.TestCase):
    """Test different FSDP wrapping strategies."""

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fsdp_by_block_strategy(self):
        """Test FSDP BY_BLOCK wrapping strategy."""
        config_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "configs"
            / "examples"
            / "debug"
            / "gpt2"
            / "gpt2_debug_fsdp.yaml"
        )

        if not config_path.exists():
            self.skipTest(f"Config not found: {config_path}")

        trainer_config = loader.load_training_config(config_path, gpt.GPT2Config)

        # Verify BY_BLOCK strategy is configured
        self.assertEqual(trainer_config.training.execution.fsdp.wrapping_strategy, "BY_BLOCK")

        # Create model
        model = gpt2.GPT2.from_config(trainer_config.llm)

        # Get FSDP wrap policy
        wrap_policy = model.get_fsdp_wrap_policy(execution_configs.FSDPWrapStrategy.BY_BLOCK)
        self.assertIsNotNone(wrap_policy)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fsdp_size_based_strategy(self):
        """Test FSDP SIZE_BASED wrapping strategy."""
        config_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "configs"
            / "examples"
            / "debug"
            / "llama3"
            / "llama3_debug_fsdp.yaml"
        )

        if not config_path.exists():
            self.skipTest(f"Config not found: {config_path}")

        trainer_config = loader.load_training_config(config_path, llama.Llama3Config)

        # Create model
        model = llama3.Llama3.from_config(trainer_config.llm)

        # Get FSDP wrap policy with size-based strategy
        wrap_policy = model.get_fsdp_wrap_policy(execution_configs.FSDPWrapStrategy.SIZE_BASED)
        self.assertIsNotNone(wrap_policy)


if __name__ == "__main__":
    unittest.main()
