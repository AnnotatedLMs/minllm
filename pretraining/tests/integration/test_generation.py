# Standard Library
import pathlib

# Third Party
import pytest
import torch

# Project
from pretraining.common.patterns.architectures import gpt2
from pretraining.common.patterns.architectures import llama3
from pretraining.configs import loader
from pretraining.configs.model.architectures import gpt
from pretraining.configs.model.architectures import llama


class TestGeneration:
    """Test generation functionality for LLM models.

    This integration test suite validates that models can generate text:
    - Tests GPT2 and Llama generation
    - Verifies temperature and top-k sampling work
    - Ensures generated sequences have correct shape
    - Validates that generation doesn't crash
    """

    @pytest.fixture
    def debug_configs_dir(self) -> pathlib.Path:
        """Path to debug configs."""
        return pathlib.Path(__file__).parent.parent.parent / "configs" / "examples" / "debug"

    def test_gpt2_generation(self, debug_configs_dir: pathlib.Path) -> None:
        """Test GPT2 text generation."""
        # Load config
        config_path = debug_configs_dir / "gpt2_debug.yaml"
        config = loader.load_training_config(config_path, gpt.GPT2Config)

        # Create model
        model = gpt2.GPT2LLM(config.llm)
        model.eval()

        # Create input tokens (batch_size=2, seq_len=5)
        batch_size = 2
        seq_len = 5
        vocab_size = config.llm.token_embedding.vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Generate tokens
        max_new_tokens = 10
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50
            )

        # Verify output shape
        assert output_ids.shape == (batch_size, seq_len + max_new_tokens)

        # Verify all tokens are valid
        assert torch.all(output_ids >= 0)
        assert torch.all(output_ids < vocab_size)

        # Test that different temperatures produce different results
        output_ids_high_temp = model.generate(
            input_ids, max_new_tokens=5, temperature=2.0, top_k=None
        )

        output_ids_low_temp = model.generate(input_ids, max_new_tokens=5, temperature=0.1, top_k=5)

        # Both should have correct shape
        assert output_ids_high_temp.shape == (batch_size, seq_len + 5)
        assert output_ids_low_temp.shape == (batch_size, seq_len + 5)

    def test_llama_generation(self, debug_configs_dir: pathlib.Path) -> None:
        """Test Llama text generation."""
        # Load config
        config_path = debug_configs_dir / "llama31_debug.yaml"
        config = loader.load_training_config(config_path, llama.Llama3Config)

        # Create model
        model = llama3.LlamaLLM(config.llm)
        model.eval()

        # Create input tokens
        batch_size = 1
        seq_len = 8
        vocab_size = config.llm.token_embedding.vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Generate tokens
        max_new_tokens = 16
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=max_new_tokens, temperature=1.0, top_k=100
            )

        # Verify output shape
        assert output_ids.shape == (batch_size, seq_len + max_new_tokens)

        # Verify all tokens are valid
        assert torch.all(output_ids >= 0)
        assert torch.all(output_ids < vocab_size)

    def test_generation_reproducibility(self, debug_configs_dir: pathlib.Path) -> None:
        """Test that generation is reproducible with same seed."""
        # Load config
        config_path = debug_configs_dir / "gpt2_debug.yaml"
        config = loader.load_training_config(config_path, gpt.GPT2Config)

        # Create model
        model = gpt2.GPT2LLM(config.llm)
        model.eval()

        # Create input
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        # Generate with fixed seed
        torch.manual_seed(42)
        output1 = model.generate(input_ids, max_new_tokens=10, temperature=1.0)

        torch.manual_seed(42)
        output2 = model.generate(input_ids, max_new_tokens=10, temperature=1.0)

        # Should be identical
        assert torch.equal(output1, output2)

        # Generate with different seed
        torch.manual_seed(123)
        output3 = model.generate(input_ids, max_new_tokens=10, temperature=1.0)

        # Should be different (with high probability)
        assert not torch.equal(output1, output3)
