"""Tests for YAML config parsing."""

# Standard Library
import pathlib
import tempfile

# Third Party
import pytest
import yaml

# Project
from pretraining.configs.llm import llm_configs
from pretraining.configs.llm import weight_init
from pretraining.configs.parsers.model_parser import ModelConfigParser
from pretraining.configs.parsers.training_parser import TrainingConfigParser
from pretraining.configs.registry import ConfigRegistry


class TestConfigParsing:
    """Test parsing of YAML configurations."""

    @pytest.fixture
    def registry(self):
        """Create a config registry for testing."""
        model_parser = ModelConfigParser()
        training_parser = TrainingConfigParser()
        return ConfigRegistry(model_parser, training_parser)

    @pytest.fixture
    def example_configs_dir(self):
        """Path to example configs."""
        return pathlib.Path(__file__).parent.parent / "examples"

    def test_parse_gpt2_config(self, registry, example_configs_dir):
        """Test parsing GPT-2 configuration."""
        config_path = example_configs_dir / "gpt2.yaml"
        config = registry.parse_config(config_path)

        # Check it parsed to correct type
        assert isinstance(config.llm, llm_configs.GPT2Config)

        # Check model dimensions
        assert config.llm.transformer_config.hidden_dim == 768
        assert config.llm.transformer_config.n_layers == 12
        assert config.llm.transformer_config.vocab_size == 50257

        # Check weight init
        assert config.llm.weight_init_config.strategy == "gpt2"
        assert config.llm.weight_init_config.std == 0.02
        assert config.llm.weight_init_config.residual_pattern == "c_proj.weight"

        # Check position embeddings exist
        assert config.llm.position_embedding_config is not None
        assert config.llm.position_embedding_config.max_position_embeddings == 1024

    def test_parse_llama_config(self, registry, example_configs_dir):
        """Test parsing Llama 3.1 configuration."""
        config_path = example_configs_dir / "llama31.yaml"
        config = registry.parse_config(config_path)

        # Check it parsed to correct type
        assert isinstance(config.llm, llm_configs.LlamaConfig)

        # Check model dimensions
        assert config.llm.transformer_config.hidden_dim == 4096
        assert config.llm.transformer_config.n_layers == 32
        assert config.llm.transformer_config.vocab_size == 128256

        # Check weight init
        assert config.llm.weight_init_config.strategy == "pytorch_default"

        # Check no position embeddings
        assert not hasattr(config.llm, "position_embedding_config")

        # Check RoPE config
        assert config.llm.transformer_config.rope_config is not None
        assert config.llm.transformer_config.rope_config.theta == 500000.0

        # Check GQA
        attention_config = config.llm.transformer_config.attention_config
        assert hasattr(attention_config, "num_kv_heads")
        assert attention_config.num_kv_heads == 8

    def test_parse_deepseek_config(self, registry, example_configs_dir):
        """Test parsing DeepSeek3 configuration."""
        config_path = example_configs_dir / "deepseek3.yaml"
        config = registry.parse_config(config_path)

        # Check it parsed to correct type
        assert isinstance(config.llm, llm_configs.DeepSeekConfig)

        # Check model dimensions
        assert config.llm.transformer_config.hidden_dim == 7168
        assert config.llm.transformer_config.n_layers == 61

        # Check weight init
        assert config.llm.weight_init_config.strategy == "pytorch_default"

        # Check MoE config
        assert config.llm.transformer_config.moe_config is not None
        assert config.llm.transformer_config.moe_config.num_experts == 16
        assert config.llm.transformer_config.moe_config.num_experts_per_token == 2

        # Check MTP config
        assert config.llm.mtp_config is not None
        assert config.llm.mtp_config.n_predict == 3

        # Check MLA attention
        attention_config = config.llm.transformer_config.attention_config
        assert hasattr(attention_config, "kv_compression_dim")
        assert attention_config.kv_compression_dim == 512

    def test_missing_required_field(self, registry):
        """Test that missing required fields raise appropriate errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Missing weight_init section
            yaml.dump(
                {
                    "model": {
                        "architecture": "gpt2",
                        "token_embedding": {
                            "vocab_size": 100,
                            "embedding_dim": 64,
                            "embedding_dropout": 0.0,
                            "init_std": 0.02,
                        },
                        # weight_init missing!
                    },
                    "training": {"max_iters": 100},  # Minimal training config
                },
                f,
            )
            f.flush()

            with pytest.raises(ValueError, match="weight_init"):
                registry.parse_config(f.name)

    def test_invalid_weight_init_strategy(self, registry):
        """Test that invalid weight init strategy raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "model": {
                        "architecture": "gpt2",
                        "weight_init": {
                            "strategy": "invalid_strategy"  # Invalid!
                        },
                        # ... other required fields
                    }
                },
                f,
            )
            f.flush()

            with pytest.raises(ValueError, match="Unknown initialization strategy"):
                registry.parse_config(f.name)

    def test_weight_init_validation(self):
        """Test weight init config validation."""
        # GPT-2 strategy requires additional fields
        with pytest.raises(ValueError, match="std is required"):
            weight_init.InitializationConfig(strategy="gpt2")

        # PyTorch default needs no additional fields
        config = weight_init.InitializationConfig(strategy="pytorch_default")
        assert config.std is None
        assert config.residual_pattern is None

    def test_architecture_config_matching(self, registry):
        """Test that architecture determines correct config type."""
        test_cases = [
            ("gpt2", llm_configs.GPT2Config),
            ("llama3.1", llm_configs.LlamaConfig),
            ("deepseek3", llm_configs.DeepSeekConfig),
        ]

        for arch, expected_type in test_cases:
            # Create minimal config for each architecture
            config_dict = self._create_minimal_config(arch)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(config_dict, f)
                f.flush()

                config = registry.parse_config(f.name)
                assert isinstance(config.llm, expected_type)

    def _create_minimal_config(self, architecture: str) -> dict:
        """Create minimal valid config for testing."""
        base_config = {
            "model": {
                "architecture": architecture,
                "weight_init": {
                    "strategy": "gpt2" if architecture == "gpt2" else "pytorch_default",
                },
                "token_embedding": {
                    "vocab_size": 100,
                    "embedding_dim": 64,
                    "embedding_dropout": 0.0,
                    "init_std": 0.02,
                },
                "transformer_backbone": {
                    "vocab_size": 100,
                    "hidden_dim": 64,
                    "n_layers": 2,
                    "block_size": 128,
                    "dropout": 0.0,
                    "bias": True,
                    "normalization": {"norm_type": "layer_norm", "norm_eps": 1e-5, "bias": True},
                    "attention": {
                        "num_heads": 4,
                        "dropout": 0.0,
                        "bias": True,
                        "max_seq_length": 128,
                        "is_causal": True,
                    },
                    "ffn": {
                        "ffn_type": "mlp",
                        "intermediate_dim": 256,
                        "activation": "gelu",
                        "dropout": 0.0,
                        "bias": True,
                    },
                },
                "output_head": {"tie_word_embeddings": True, "lm_head_bias": False},
            },
            "training": {
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_iters": 100,
                "log_interval": 10,
                "lr_warmup_iters": 10,
                "lr_decay_iters": 100,
                "min_lr": 1e-5,
                "eval_interval": 50,
                "eval_iters": 10,
                "seed": 42,
            },
            "optimizer": {
                "name": "adamw",
                "lr": 1e-4,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.95,
                "grad_clip": 1.0,
            },
            "lr_schedule": {
                "schedule_type": "cosine",
                "warmup_iters": 10,
                "lr_decay_iters": 100,
                "min_lr": 1e-5,
            },
            "checkpointing": {"save_dir": "./test_checkpoints", "save_interval": 50},
            "data": {
                "dataset_name": "test",
                "data_dir": "./test_data",
                "train_split": 0.9,
                "val_split": 0.1,
            },
            "system": {
                "device": "cpu",
                "torch_dtype": "float32",
                "compile": False,
                "distributed": False,
            },
            "logging": {"use_wandb": False, "wandb_project": "test"},
        }

        # Add architecture-specific fields
        if architecture == "gpt2":
            base_config["model"]["weight_init"].update(
                {"std": 0.02, "residual_pattern": "c_proj.weight", "position_init_std": 0.02}
            )
            base_config["model"]["position_embedding"] = {
                "max_position_embeddings": 128,
                "embedding_dim": 64,
                "init_std": 0.02,
            }
        elif architecture == "llama3.1":
            base_config["model"]["transformer_backbone"]["attention"]["num_kv_heads"] = 2
            base_config["model"]["transformer_backbone"]["rope"] = {"theta": 10000.0, "dim": 16}
        elif architecture == "deepseek3":
            base_config["model"]["transformer_backbone"]["attention"].update(
                {
                    "head_dim": 16,
                    "kv_compression_dim": 32,
                    "query_compression_dim": 32,
                    "rope_dim": 16,
                }
            )
            base_config["model"]["transformer_backbone"]["rope"] = {"theta": 10000.0, "dim": 16}
            base_config["model"]["transformer_backbone"]["moe"] = {
                "num_experts": 4,
                "num_experts_per_token": 2,
            }
            base_config["model"]["mtp"] = {
                "n_predict": 2,
                "prediction_depth": 1,
                "dropout_rate": 0.1,
            }
            # Remove FFN for MoE
            del base_config["model"]["transformer_backbone"]["ffn"]

        return base_config
