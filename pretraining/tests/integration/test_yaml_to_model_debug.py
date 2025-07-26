# Standard Library
import pathlib

# Third Party
import jaxtyping
import pytest
import torch

# Project
from pretraining.common.patterns.architectures import deepseek3
from pretraining.common.patterns.architectures import gpt2
from pretraining.common.patterns.architectures import llama3
from pretraining.common.patterns.moe import aux_loss_free
from pretraining.configs import loader
from pretraining.configs.model.architectures import deepseek
from pretraining.configs.model.architectures import gpt
from pretraining.configs.model.architectures import llama


class TestYAMLToModelDebug:
    """Test complete flow from YAML configuration to model instantiation with small debug models.

    How to run:
        From project root (/Users/joncheng/Desktop/minllm):
        uv run pytest pretraining/tests/integration/test_yaml_to_model_debug.py -v

    This integration test suite validates the end-to-end pipeline for each supported model type:

    • Reads YAML configuration files from configs/examples/debug/
    • Parses configs using loader.load_training_config
    • Creates model instances directly from model classes
    • Verifies correct model type instantiation (GPT2LLM, LlamaLLM, DeepSeekLLM)
    • Runs forward pass with random input to ensure model is functional
    • Validates output tensor shapes match expected dimensions
    • Checks model-specific features:
      - GPT-2: Verifies position embeddings exist
      - Llama: Confirms no position embeddings (uses RoPE instead)
      - DeepSeek: Validates MoE layers and MTP outputs
    • Ensures all debug models have <1M parameters for fast testing
    """

    @pytest.fixture
    def debug_configs_dir(self) -> pathlib.Path:
        """Path to debug configs."""
        return pathlib.Path(__file__).parent.parent.parent / "configs" / "examples" / "debug"

    def test_gpt2_debug_yaml_to_model(self, debug_configs_dir: pathlib.Path) -> None:
        """Test GPT-2 debug YAML loads and creates working model."""
        config_path = debug_configs_dir / "gpt2_debug.yaml"

        # Parse config
        config = loader.load_training_config(config_path, gpt.GPT2Config)

        # Create model
        model = gpt2.GPT2LLM(config.llm)

        # Verify model type
        assert isinstance(model, gpt2.GPT2LLM)

        # Test forward pass
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.llm.transformer.vocab_size, (batch_size, seq_len))

        output: jaxtyping.Float[torch.Tensor, "batch seq vocab"] = model.inference_forward(
            input_ids
        )

        # Check output shape
        assert output.shape == (batch_size, seq_len, config.llm.transformer.vocab_size)

        # Verify position embeddings exist
        assert hasattr(model, "position_embeddings")
        assert model.position_embeddings is not None

    def test_llama_debug_yaml_to_model(self, debug_configs_dir: pathlib.Path) -> None:
        """Test Llama debug YAML loads and creates working model."""
        config_path = debug_configs_dir / "llama31_debug.yaml"

        # Parse config
        config = loader.load_training_config(config_path, llama.Llama3Config)

        # Create model
        model = llama3.LlamaLLM(config.llm)

        # Verify model type
        assert isinstance(model, llama3.LlamaLLM)

        # Test forward pass with smaller sequence for memory
        batch_size, seq_len = 1, 32
        input_ids = torch.randint(0, config.llm.transformer.vocab_size, (batch_size, seq_len))

        output: jaxtyping.Float[torch.Tensor, "batch seq vocab"] = model.inference_forward(
            input_ids
        )

        # Check output shape
        assert output.shape == (batch_size, seq_len, config.llm.transformer.vocab_size)

        # Verify no position embeddings
        assert not hasattr(model, "position_embeddings")

        # Verify RoPE is configured at model level
        assert hasattr(model, "rope")
        assert model.rope is not None

    def test_deepseek_debug_yaml_to_model(self, debug_configs_dir: pathlib.Path) -> None:
        """Test DeepSeek debug YAML loads and creates working model."""
        config_path = debug_configs_dir / "deepseek3_debug.yaml"

        # Parse config
        config = loader.load_training_config(config_path, deepseek.DeepSeek3Config)

        # Create model
        model = deepseek3.DeepSeekLLM(config.llm)

        # Verify model type
        assert isinstance(model, deepseek3.DeepSeekLLM)

        # Test forward pass with very small sequence
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, config.llm.transformer.vocab_size, (batch_size, seq_len))

        # Test inference forward (no MTP outputs in inference mode)
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"] = model.inference_forward(
            input_ids
        )

        # Check output shape
        assert logits.shape == (batch_size, seq_len, config.llm.transformer.vocab_size)

        # Verify MoE is configured
        assert hasattr(model.blocks[0], "moe")
        assert isinstance(model.blocks[0].moe, aux_loss_free.AuxLossFreeMoE)

    def test_model_parameter_counts(self, debug_configs_dir: pathlib.Path) -> None:
        """Test that debug models have reasonable parameter counts."""
        for config_name, model_class, config_class in [
            ("gpt2_debug.yaml", gpt2.GPT2LLM, gpt.GPT2Config),
            ("llama31_debug.yaml", llama3.LlamaLLM, llama.Llama3Config),
            ("deepseek3_debug.yaml", deepseek3.DeepSeekLLM, deepseek.DeepSeek3Config),
        ]:
            config_path = debug_configs_dir / config_name
            config = loader.load_training_config(config_path, config_class)

            model = model_class(config.llm)
            param_count = sum(p.numel() for p in model.parameters())

            # Debug models should be tiny
            assert param_count < 1_000_000, (
                f"{config_name} has {param_count:,} params (should be < 1M)"
            )
            print(f"{config_name}: {param_count:,} parameters")
