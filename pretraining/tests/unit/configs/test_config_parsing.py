"""Tests for YAML config parsing with Pydantic."""

# Standard Library
import pathlib

# Third Party
import pytest

# Project
from pretraining.configs import core
from pretraining.configs import loader
from pretraining.configs.model import initialization
from pretraining.configs.model import transformer
from pretraining.configs.model.architectures import deepseek
from pretraining.configs.model.architectures import gpt
from pretraining.configs.model.architectures import llama
from pretraining.configs.model.components import attention
from pretraining.configs.model.components import embeddings
from pretraining.configs.model.components import feedforward
from pretraining.configs.model.components import heads
from pretraining.configs.model.components import normalization


class TestConfigParsing:
    """Test parsing of YAML configurations."""

    @pytest.fixture
    def example_configs_dir(self):
        """Path to example configs."""
        # Configs are in pretraining/configs/examples/debug
        return pathlib.Path(__file__).parent.parent.parent.parent / "configs" / "examples" / "debug"

    def test_parse_gpt2_config(self, example_configs_dir):
        """Test parsing GPT-2 configuration."""
        config_path = example_configs_dir / "gpt2_debug_cpu.yaml"
        config = loader.load_training_config(config_path, gpt.GPT2Config)

        # Check it parsed to correct type
        assert isinstance(config.llm, gpt.GPT2Config)

        # Check model dimensions
        assert config.llm.transformer.hidden_dim == 64
        assert config.llm.transformer.n_layers == 2
        assert config.llm.vocab_size == 50257  # GPT-2 vocab for FineWeb

        # Check weight init
        assert isinstance(config.llm.weight_init, initialization.GPT2InitConfig)
        assert config.llm.weight_init.std == 0.02
        assert config.llm.weight_init.residual_pattern == "down_proj.weight"

        # Check position embeddings exist
        assert config.llm.position_embedding is not None
        assert config.llm.position_embedding.max_position_embeddings == 128

        # Check training config
        assert config.training.max_iters == 10
        assert config.training.batch.batch_size == 2
        assert config.training.batch.sequence_length == 128

    def test_parse_llama_config(self, example_configs_dir):
        """Test parsing Llama 3.1 configuration."""
        config_path = example_configs_dir / "llama3_debug_cpu.yaml"
        config = loader.load_training_config(config_path, llama.Llama3Config)

        # Check it parsed to correct type
        assert isinstance(config.llm, llama.Llama3Config)

        # Check model dimensions
        assert config.llm.transformer.hidden_dim == 64
        assert config.llm.transformer.n_layers == 2
        assert config.llm.vocab_size == 50257  # GPT-2 vocab for FineWeb

        # Check weight init (Llama uses default, so None)
        assert config.llm.weight_init is None

        # Check no position embeddings
        assert not hasattr(config.llm, "position_embedding")

        # Check RoPE config
        assert config.llm.transformer.rope is not None
        assert config.llm.transformer.rope.theta == 10000.0

        # Check GQA
        attention_config = config.llm.transformer.attention
        assert hasattr(attention_config, "num_kv_heads")
        assert attention_config.num_kv_heads == 2

    def test_parse_deepseek_config(self, example_configs_dir):
        """Test parsing DeepSeek3 configuration."""
        config_path = example_configs_dir / "deepseek3_debug_cpu.yaml"
        config = loader.load_training_config(config_path, deepseek.DeepSeek3Config)

        # Check it parsed to correct type
        assert isinstance(config.llm, deepseek.DeepSeek3Config)

        # Check model dimensions
        assert config.llm.transformer.hidden_dim == 64
        assert config.llm.transformer.n_layers == 2

        # Check weight init (Llama uses default, so None)
        assert config.llm.weight_init is None

        # Check MoE config
        assert config.llm.transformer.moe is not None
        assert config.llm.transformer.moe.num_experts == 4
        assert config.llm.transformer.moe.num_experts_per_token == 2

        # Check MTP config
        assert config.llm.mtp is not None
        assert config.llm.mtp.n_predict == 2

        # Check MLA attention
        attention_config = config.llm.transformer.attention
        assert hasattr(attention_config, "kv_compression_dim")
        assert attention_config.kv_compression_dim == 32

        # Check architecture-specific training configs
        assert config.training.moe_training is not None

    def test_config_validation(self, example_configs_dir):
        """Test that configs are properly validated."""
        # All debug configs should load without errors
        configs = [
            ("gpt2_debug_cpu.yaml", gpt.GPT2Config),
            ("llama3_debug_cpu.yaml", llama.Llama3Config),
            ("deepseek3_debug_cpu.yaml", deepseek.DeepSeek3Config),
        ]

        for yaml_file, config_class in configs:
            config_path = example_configs_dir / yaml_file
            config = loader.load_training_config(config_path, config_class)

            # Verify key constraints are met
            assert config.llm.transformer.n_layers > 0
            assert config.llm.transformer.hidden_dim > 0
            assert config.llm.vocab_size > 0
            assert config.training.max_iters > 0
            assert config.training.batch.batch_size > 0
            assert 0 <= config.llm.transformer.dropout < 1

    def test_config_round_trip(self, example_configs_dir):
        """Test that configs can be serialized and deserialized."""
        config_path = example_configs_dir / "gpt2_debug_cpu.yaml"
        config = loader.load_training_config(config_path, gpt.GPT2Config)

        # Convert to dict and back
        config_dict = {"llm": config.llm.model_dump(), "training": config.training.model_dump()}

        # Should be able to recreate from dict
        config2 = core.TrainerConfig(
            llm=gpt.GPT2Config.model_validate(config_dict["llm"]),
            training=config.training.__class__.model_validate(config_dict["training"]),
        )

        # Key fields should match
        assert config2.llm.transformer.hidden_dim == config.llm.transformer.hidden_dim
        assert config2.training.max_iters == config.training.max_iters


class TestWeightInitConfigs:
    """Test weight initialization configurations."""

    def test_weight_init_validation(self):
        """Test weight init config validation."""

        # GPT-2 config with required fields
        config = initialization.GPT2InitConfig(
            std=0.02, residual_pattern="c_proj.weight", position_init_std=0.02
        )
        assert config.std == 0.02
        assert config.residual_pattern == "c_proj.weight"


class TestTransformerConfigs:
    """Test transformer configuration validation."""

    def test_transformer_config_ffn_moe_validation(self):
        """Test that transformer config requires either FFN or MoE but not both."""
        # Create base configs
        norm_config = normalization.LayerNormConfig(norm_eps=1e-5, bias=True)
        attn_config = attention.MultiHeadAttentionConfig(
            num_heads=4,
            bias=True,
            max_seq_length=128,
            is_causal=True,
            use_flash_attention=False,
        )
        ffn_config = feedforward.FFNConfig(intermediate_dim=256, activation="gelu", bias=True)
        moe_config = feedforward.MoEConfig(num_experts=4, num_experts_per_token=2)

        # Test valid config with FFN (using GPT2TransformerConfig)
        config = transformer.GPT2TransformerConfig(
            hidden_dim=64,
            n_layers=2,
            block_size=128,
            normalization=norm_config,
            attention=attn_config,
            ffn=ffn_config,
        )
        assert config.ffn is not None
        assert config.moe is None

        # Test valid config with MoE (using DeepSeek3TransformerConfig with MLA attention)
        mla_attn_config = attention.MultiHeadLatentAttentionConfig(
            num_heads=4,
            head_dim=16,
            kv_compression_dim=32,
            query_compression_dim=48,
            rope_dim=16,
            bias=True,
            max_seq_length=128,
            is_causal=True,
            use_flash_attention=False,
        )
        config = transformer.DeepSeek3TransformerConfig(
            hidden_dim=64,
            n_layers=2,
            block_size=128,
            normalization=norm_config,
            attention=mla_attn_config,
            moe=moe_config,
        )
        assert config.moe is not None
        assert config.ffn is None

        # Test invalid config with both FFN and MoE
        with pytest.raises(ValueError, match="Cannot have both ffn and moe"):
            transformer.DeepSeek3TransformerConfig(
                hidden_dim=64,
                n_layers=2,
                block_size=128,
                normalization=norm_config,
                attention=mla_attn_config,
                ffn=ffn_config,
                moe=moe_config,
            )

        # Test invalid config with neither FFN nor MoE
        with pytest.raises(ValueError, match="Must have either ffn or moe"):
            transformer.GPT2TransformerConfig(
                hidden_dim=64,
                n_layers=2,
                block_size=128,
                normalization=norm_config,
                attention=attn_config,
            )

    def test_attention_config_validation(self):
        """Test attention configuration validation."""
        # Test GroupedQueryAttention validation
        with pytest.raises(ValueError, match="num_kv_heads .* cannot exceed num_heads"):
            attention.GroupedQueryAttentionConfig(
                num_heads=4,
                num_kv_heads=8,  # Invalid: more KV heads than Q heads
                bias=True,
                max_seq_length=128,
                is_causal=True,
                use_flash_attention=False,
            )

        with pytest.raises(ValueError, match="num_heads .* must be divisible by num_kv_heads"):
            attention.GroupedQueryAttentionConfig(
                num_heads=7,
                num_kv_heads=3,  # Invalid: 7 not divisible by 3
                bias=True,
                max_seq_length=128,
                is_causal=True,
                use_flash_attention=False,
            )


class TestLLMConfigs:
    """Test LLM configuration validation."""

    def test_gpt2_config_validation(self):
        """Test GPT-2 specific validation."""
        # Create configs with mismatched dimensions
        token_config = embeddings.TokenEmbeddingConfig(embedding_dim=64, init_std=0.02)

        position_config = embeddings.LearnedPositionEmbeddingConfig(
            max_position_embeddings=128,
            embedding_dim=128,  # Mismatch with token embedding
            init_std=0.02,
        )

        # This should fail validation
        with pytest.raises(Exception) as exc_info:
            # Need to create a full config to trigger validation
            norm_config = normalization.LayerNormConfig(norm_eps=1e-5, bias=True)
            attn_config = attention.MultiHeadAttentionConfig(
                num_heads=4,
                bias=True,
                max_seq_length=128,
                is_causal=True,
                use_flash_attention=False,
            )
            ffn_config = feedforward.FFNConfig(intermediate_dim=256, activation="gelu", bias=True)

            transformer_config = transformer.GPT2TransformerConfig(
                hidden_dim=64,
                n_layers=2,
                block_size=128,
                normalization=norm_config,
                attention=attn_config,
                ffn=ffn_config,
            )

            gpt.GPT2Config(
                vocab_size=1000,
                token_embedding=token_config,
                position_embedding=position_config,
                transformer=transformer_config,
                output_head=heads.OutputHeadConfig(tie_word_embeddings=True, lm_head_bias=False),
                weight_init=initialization.GPT2InitConfig(
                    std=0.02, residual_pattern="c_proj", position_init_std=0.02
                ),
            )

        # Check that the error message contains the expected text
        assert "position embedding_dim" in str(exc_info.value)
        assert "must match" in str(exc_info.value)
        assert "transformer hidden_dim" in str(exc_info.value)

    def test_gpt2_requires_tied_embeddings(self):
        """Test that GPT-2 requires tied embeddings."""
        # Create a valid base config
        token_config = embeddings.TokenEmbeddingConfig(embedding_dim=64, init_std=0.02)
        position_config = embeddings.LearnedPositionEmbeddingConfig(
            max_position_embeddings=128, embedding_dim=64, init_std=0.02
        )
        norm_config = normalization.LayerNormConfig(norm_eps=1e-5, bias=True)
        attn_config = attention.MultiHeadAttentionConfig(
            num_heads=4,
            bias=True,
            max_seq_length=128,
            is_causal=True,
            use_flash_attention=False,
        )
        ffn_config = feedforward.FFNConfig(intermediate_dim=256, activation="gelu", bias=True)
        transformer_config = transformer.GPT2TransformerConfig(
            hidden_dim=64,
            n_layers=2,
            block_size=128,
            normalization=norm_config,
            attention=attn_config,
            ffn=ffn_config,
        )

        # Test untied embeddings (should fail)
        with pytest.raises(ValueError, match="GPT-2 requires tied embeddings"):
            gpt.GPT2Config(
                vocab_size=1000,
                token_embedding=token_config,
                position_embedding=position_config,
                transformer=transformer_config,
                output_head=heads.OutputHeadConfig(tie_word_embeddings=False, lm_head_bias=False),
                weight_init=initialization.GPT2InitConfig(
                    std=0.02, residual_pattern="c_proj", position_init_std=0.02
                ),
            )

        # Test lm_head_bias with tied embeddings (should fail)
        with pytest.raises(ValueError, match="GPT-2 cannot use lm_head_bias"):
            gpt.GPT2Config(
                vocab_size=1000,
                token_embedding=token_config,
                position_embedding=position_config,
                transformer=transformer_config,
                output_head=heads.OutputHeadConfig(tie_word_embeddings=True, lm_head_bias=True),
                weight_init=initialization.GPT2InitConfig(
                    std=0.02, residual_pattern="c_proj", position_init_std=0.02
                ),
            )
