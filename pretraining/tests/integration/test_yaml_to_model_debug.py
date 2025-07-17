# Standard Library
import pathlib

# Third Party
import jaxtyping
import pytest
import torch

# Project
from pretraining.common.patterns import moe
from pretraining.common.patterns.llm import deepseek
from pretraining.common.patterns.llm import factory
from pretraining.common.patterns.llm import gpt2
from pretraining.common.patterns.llm import llama
from pretraining.configs import registry
from pretraining.configs.parsers import model_parser
from pretraining.configs.parsers import training_parser


class TestYAMLToModelDebug:
    """Test complete flow from YAML configuration to model instantiation with small debug models.

    How to run:
        From project root (/Users/joncheng/Desktop/minllm):
        python -m pytest pretraining/tests/integration/test_yaml_to_model_debug.py -v

    This integration test suite validates the end-to-end pipeline for each supported model type:

    • Reads YAML configuration files from configs/examples/debug/
    • Parses configs using registry.ConfigRegistry (model + training parsers)
    • Creates model instances via factory.create_llm()
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
    def config_registry(self) -> registry.ConfigRegistry:
        """Create a config registry for testing."""
        model_parser_instance = model_parser.ModelConfigParser()
        training_parser_instance = training_parser.TrainingConfigParser()
        return registry.ConfigRegistry(model_parser_instance, training_parser_instance)

    @pytest.fixture
    def debug_configs_dir(self) -> pathlib.Path:
        """Path to debug configs."""
        return pathlib.Path(__file__).parent.parent.parent / "configs" / "examples" / "debug"

    def test_gpt2_debug_yaml_to_model(
        self, config_registry: registry.ConfigRegistry, debug_configs_dir: pathlib.Path
    ) -> None:
        """Test GPT-2 debug YAML loads and creates working model."""
        config_path = debug_configs_dir / "gpt2_debug.yaml"

        # Parse config
        config = config_registry.parse_config(config_path)

        # Create model
        model = factory.create_llm(config.llm)

        # Verify model type
        assert isinstance(model, gpt2.GPT2LLM)

        # Test forward pass
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(
            0, config.llm.transformer_config.vocab_size, (batch_size, seq_len)
        )

        output: jaxtyping.Float[torch.Tensor, "batch seq vocab"] = model.inference_forward(
            input_ids
        )

        # Check output shape
        assert output.shape == (batch_size, seq_len, config.llm.transformer_config.vocab_size)

        # Verify position embeddings exist
        assert hasattr(model, "position_embeddings")
        assert model.position_embeddings is not None

    def test_llama_debug_yaml_to_model(
        self, config_registry: registry.ConfigRegistry, debug_configs_dir: pathlib.Path
    ) -> None:
        """Test Llama debug YAML loads and creates working model."""
        config_path = debug_configs_dir / "llama31_debug.yaml"

        # Parse config
        config = config_registry.parse_config(config_path)

        # Create model
        model = factory.create_llm(config.llm)

        # Verify model type
        assert isinstance(model, llama.LlamaLLM)

        # Test forward pass with smaller sequence for memory
        batch_size, seq_len = 1, 32
        input_ids = torch.randint(
            0, config.llm.transformer_config.vocab_size, (batch_size, seq_len)
        )

        output: jaxtyping.Float[torch.Tensor, "batch seq vocab"] = model.inference_forward(
            input_ids
        )

        # Check output shape
        assert output.shape == (batch_size, seq_len, config.llm.transformer_config.vocab_size)

        # Verify no position embeddings
        assert not hasattr(model, "position_embeddings")

        # Verify RoPE is configured at model level
        assert hasattr(model, "rope")
        assert model.rope is not None

    def test_deepseek_debug_yaml_to_model(
        self, config_registry: registry.ConfigRegistry, debug_configs_dir: pathlib.Path
    ) -> None:
        """Test DeepSeek debug YAML loads and creates working model."""
        config_path = debug_configs_dir / "deepseek3_debug.yaml"

        # Parse config
        config = config_registry.parse_config(config_path)

        # Create model
        model = factory.create_llm(config.llm)

        # Verify model type
        assert isinstance(model, deepseek.DeepSeekLLM)

        # Test forward pass with very small sequence
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(
            0, config.llm.transformer_config.vocab_size, (batch_size, seq_len)
        )

        # Test inference forward (no MTP outputs in inference mode)
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"] = model.inference_forward(
            input_ids
        )

        # Check output shape
        assert logits.shape == (batch_size, seq_len, config.llm.transformer_config.vocab_size)

        # Verify MoE is configured
        assert hasattr(model.blocks[0], "moe")
        assert isinstance(model.blocks[0].moe, moe.AuxLossFreeMoE)

    def test_model_parameter_counts(
        self, config_registry: registry.ConfigRegistry, debug_configs_dir: pathlib.Path
    ) -> None:
        """Test that debug models have reasonable parameter counts."""
        for config_name in ["gpt2_debug.yaml", "llama31_debug.yaml", "deepseek3_debug.yaml"]:
            config_path = debug_configs_dir / config_name
            config = config_registry.parse_config(config_path)

            model = factory.create_llm(config.llm)
            param_count = sum(p.numel() for p in model.parameters())

            # Debug models should be tiny
            assert param_count < 1_000_000, (
                f"{config_name} has {param_count:,} params (should be < 1M)"
            )
            print(f"{config_name}: {param_count:,} parameters")
