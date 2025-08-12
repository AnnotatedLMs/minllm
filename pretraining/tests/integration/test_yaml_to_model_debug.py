# Standard Library
import pathlib

# Third Party
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
    • Verifies correct model type instantiation (GPT2, Llama3, DeepSeek3)
    • Runs forward pass with random input to ensure model is functional
    • Validates output tensor shapes match expected dimensions
    • Checks model-specific features:
      - GPT-2: Verifies position embeddings exist
      - Llama: Confirms no position embeddings (uses RoPE instead)
      - DeepSeek: Validates MoE layers and MTP outputs
    • Ensures all debug models have <1M parameters for fast testing
    """

    def test_gpt2_debug_yaml_to_model(self, debug_configs_dir: pathlib.Path) -> None:
        """Test GPT-2 debug YAML loads and creates working model."""
        config_path = debug_configs_dir / "gpt2" / "gpt2_debug_cpu.yaml"

        # Parse config
        config = loader.load_training_config(config_path, gpt.GPT2Config)

        # Create model
        model = gpt2.GPT2.from_config(config.llm)

        # Verify model type
        assert isinstance(model, gpt2.GPT2)

        # Test forward pass
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.llm.vocab_size, (batch_size, seq_len))

        # Put model in eval mode and forward
        model.eval()
        with torch.no_grad():
            output = model.forward(input_ids=input_ids)

        # Check output shape
        assert output.logits.shape == (batch_size, seq_len, config.llm.vocab_size)

        # Verify position embeddings exist
        assert hasattr(model, "position_embeddings")
        assert model.position_embeddings is not None

    def test_llama_debug_yaml_to_model(self, debug_configs_dir: pathlib.Path) -> None:
        """Test Llama debug YAML loads and creates working model."""
        config_path = debug_configs_dir / "llama3" / "llama3_debug_cpu.yaml"

        # Parse config
        config = loader.load_training_config(config_path, llama.Llama3Config)

        # Create model
        model = llama3.Llama3.from_config(config.llm)

        # Verify model type
        assert isinstance(model, llama3.Llama3)

        # Test forward pass with smaller sequence for memory
        batch_size, seq_len = 1, 32
        input_ids = torch.randint(0, config.llm.vocab_size, (batch_size, seq_len))

        # Put model in eval mode and forward
        model.eval()
        with torch.no_grad():
            output = model.forward(input_ids=input_ids)

        # Check output shape
        assert output.logits.shape == (batch_size, seq_len, config.llm.vocab_size)

        # Verify no position embeddings
        assert not hasattr(model, "position_embeddings")

        # Verify RoPE is configured at model level
        assert hasattr(model, "rope")
        assert model.rope is not None

    def test_deepseek_debug_yaml_to_model(self, debug_configs_dir: pathlib.Path) -> None:
        """Test DeepSeek debug YAML loads and creates working model."""
        config_path = debug_configs_dir / "deepseek3" / "deepseek3_debug_cpu.yaml"

        # Parse config
        config = loader.load_training_config(config_path, deepseek.DeepSeek3Config)

        # Create model
        model = deepseek3.DeepSeek3.from_config(config.llm)

        # Verify model type
        assert isinstance(model, deepseek3.DeepSeek3)

        # Test forward pass with very small sequence
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, config.llm.vocab_size, (batch_size, seq_len))

        # Test inference forward (no MTP outputs in inference mode)
        model.eval()
        with torch.no_grad():
            output = model.forward(input_ids=input_ids)

        # Check output shape
        assert output.logits.shape == (batch_size, seq_len, config.llm.vocab_size)
        # In eval mode, MTP logits should not be computed
        assert output.mtp_logits is None

        # Verify MoE is configured
        assert hasattr(model.blocks[0], "moe")
        assert isinstance(model.blocks[0].moe, aux_loss_free.AuxLossFreeMoE)

    def test_model_parameter_counts(self, debug_configs_dir: pathlib.Path) -> None:
        """Test that debug models have reasonable parameter counts."""
        for config_name, model_class, config_class in [
            ("gpt2/gpt2_debug_cpu.yaml", gpt2.GPT2, gpt.GPT2Config),
            ("llama3/llama3_debug_cpu.yaml", llama3.Llama3, llama.Llama3Config),
            ("deepseek3/deepseek3_debug_cpu.yaml", deepseek3.DeepSeek3, deepseek.DeepSeek3Config),
        ]:
            config_path = debug_configs_dir / config_name
            config = loader.load_training_config(config_path, config_class)

            model = model_class.from_config(config.llm)
            param_count = sum(p.numel() for p in model.parameters())

            # Debug models should be tiny
            assert param_count < 1_000_000, (
                f"{config_name} has {param_count:,} params (should be < 1M)"
            )
            print(f"{config_name}: {param_count:,} parameters")
