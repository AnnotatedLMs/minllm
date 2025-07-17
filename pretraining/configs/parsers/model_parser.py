# Standard Library
import typing

# Project
from pretraining.configs.llm import enums
from pretraining.configs.llm import head_configs
from pretraining.configs.llm import llm_configs
from pretraining.configs.llm import position_configs as llm_position_configs
from pretraining.configs.llm import token_configs
from pretraining.configs.llm import weight_init

# Local
from pretraining.configs.transformer import attention_configs
from pretraining.configs.transformer import ffn_configs
from pretraining.configs.transformer import moe_configs
from pretraining.configs.transformer import normalization_configs
from pretraining.configs.transformer import position_configs
from pretraining.configs.transformer import transformer_configs


class ModelConfigParser:
    """Parses model-related configuration components from dictionaries."""

    def parse_attention_config(
        self, attention_dict: typing.Dict[str, typing.Any]
    ) -> attention_configs.BaseAttentionConfig:
        """Parse attention configuration based on attention_type."""
        attention_type = attention_dict.get("attention_type", "multi_head")

        # Common parameters for all attention types
        base_params = {
            "num_heads": attention_dict["num_heads"],
            "dropout": attention_dict["dropout"],
            "bias": attention_dict["bias"],
            "max_seq_length": attention_dict["max_seq_length"],
            "is_causal": attention_dict["is_causal"],
            "use_flash_attention": attention_dict["use_flash_attention"],
        }

        if attention_type == "multi_head":
            return attention_configs.MultiHeadAttentionConfig(
                **base_params, attention_type=attention_type
            )

        elif attention_type == "grouped_query":
            # GQA requires num_kv_heads
            if "num_kv_heads" not in attention_dict:
                raise ValueError("grouped_query attention requires num_kv_heads")

            return attention_configs.GroupedQueryAttentionConfig(
                **base_params,
                attention_type=attention_type,
                num_kv_heads=attention_dict["num_kv_heads"],
            )

        elif attention_type == "multi_head_latent":
            # MLA requires several additional fields
            required_fields = [
                "head_dim",
                "kv_compression_dim",
                "query_compression_dim",
                "rope_dim",
            ]
            for field in required_fields:
                if field not in attention_dict:
                    raise ValueError(f"multi_head_latent attention requires {field}")

            return attention_configs.MultiHeadLatentAttentionConfig(
                **base_params,
                attention_type=attention_type,
                head_dim=attention_dict["head_dim"],
                kv_compression_dim=attention_dict["kv_compression_dim"],
                query_compression_dim=attention_dict["query_compression_dim"],
                rope_dim=attention_dict["rope_dim"],
            )

        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

    def parse_ffn_config(self, ffn_dict: typing.Dict[str, typing.Any]) -> ffn_configs.FFNConfig:
        """Parse feedforward network configuration."""
        return ffn_configs.FFNConfig(
            ffn_type=ffn_dict["ffn_type"],
            activation=ffn_dict["activation"],
            intermediate_dim=ffn_dict.get(
                "intermediate_dim"
            ),  # Optional - calculated from expansion_factor if not provided
            expansion_factor=ffn_dict.get(
                "expansion_factor"
            ),  # Optional - one of intermediate_dim or expansion_factor must be provided
            ffn_dim_multiplier=ffn_dict.get("ffn_dim_multiplier"),  # Optional
            multiple_of=ffn_dict.get("multiple_of"),  # Optional
            dropout=ffn_dict["dropout"],
            bias=ffn_dict["bias"],
        )

    def parse_moe_config(self, moe_dict: typing.Dict[str, typing.Any]) -> moe_configs.MoEConfig:
        """Parse Mixture of Experts configuration."""
        return moe_configs.MoEConfig(
            num_experts=moe_dict["num_experts"],
            num_experts_per_token=moe_dict["num_experts_per_token"],
            expert_intermediate_dim=moe_dict.get("expert_intermediate_dim"),  # Optional
            moe_type=moe_dict["moe_type"],
            shared_expert_ratio=moe_dict["shared_expert_ratio"],
            bias_update_speed=moe_dict["bias_update_speed"],
            gate_noise_scale=moe_dict["gate_noise_scale"],
        )

    def parse_normalization_config(
        self, norm_dict: typing.Dict[str, typing.Any]
    ) -> normalization_configs.NormalizationConfig:
        """Parse normalization configuration."""
        return normalization_configs.NormalizationConfig(
            norm_type=norm_dict["norm_type"],
            norm_eps=norm_dict["norm_eps"],
            bias=norm_dict["bias"],
        )

    def parse_rope_scaling_config(
        self, scaling_dict: typing.Dict[str, typing.Any]
    ) -> position_configs.RoPEScalingConfig:
        """Parse RoPE scaling configuration."""
        return position_configs.RoPEScalingConfig(
            scale_factor=scaling_dict["scale_factor"],
            low_freq_factor=scaling_dict["low_freq_factor"],
            high_freq_factor=scaling_dict["high_freq_factor"],
            original_context_len=scaling_dict["original_context_len"],
        )

    def parse_rope_config(
        self, rope_dict: typing.Dict[str, typing.Any]
    ) -> position_configs.RoPEConfig:
        """Parse Rotary Position Embedding configuration."""
        # Parse scaling config if present
        scaling_config = None
        if "scaling" in rope_dict:
            scaling_config = self.parse_rope_scaling_config(rope_dict["scaling"])

        return position_configs.RoPEConfig(
            theta=rope_dict["theta"],
            dim=rope_dict.get("dim"),  # Optional - None means use full head_dim
            scaling=scaling_config,
        )

    def parse_mtp_config(
        self, mtp_dict: typing.Dict[str, typing.Any]
    ) -> head_configs.MultiTokenPredictionConfig:
        """Parse multi-token prediction configuration."""
        return head_configs.MultiTokenPredictionConfig(
            n_predict=mtp_dict["n_predict"],
            prediction_depth=mtp_dict["prediction_depth"],
            dropout_rate=mtp_dict["dropout_rate"],
        )

    def parse_token_embedding_config(
        self, token_dict: typing.Dict[str, typing.Any]
    ) -> token_configs.TokenEmbeddingConfig:
        """Parse token embedding configuration."""
        return token_configs.TokenEmbeddingConfig(
            vocab_size=token_dict["vocab_size"],
            embedding_dim=token_dict["embedding_dim"],
            embedding_dropout=token_dict["embedding_dropout"],
            init_std=token_dict["init_std"],
        )

    def parse_position_embedding_config(
        self, pos_dict: typing.Dict[str, typing.Any]
    ) -> llm_position_configs.LearnedPositionEmbeddingConfig:
        """Parse learned position embedding configuration."""
        return llm_position_configs.LearnedPositionEmbeddingConfig(
            max_position_embeddings=pos_dict["max_position_embeddings"],
            embedding_dim=pos_dict["embedding_dim"],
            init_std=pos_dict["init_std"],
        )

    def parse_output_head_config(
        self, head_dict: typing.Dict[str, typing.Any]
    ) -> head_configs.OutputHeadConfig:
        """Parse output head configuration."""
        return head_configs.OutputHeadConfig(
            tie_word_embeddings=head_dict["tie_word_embeddings"],
            lm_head_bias=head_dict["lm_head_bias"],
        )

    def parse_weight_init_config(
        self, init_dict: typing.Dict[str, typing.Any]
    ) -> weight_init.InitializationConfig:
        """Parse weight initialization configuration."""
        strategy = init_dict["strategy"]

        if strategy == "gpt2":
            return weight_init.InitializationConfig(
                strategy=strategy,
                std=init_dict["std"],
                residual_pattern=init_dict["residual_pattern"],
                position_init_std=init_dict["position_init_std"],
            )
        elif strategy == "pytorch_default":
            return weight_init.InitializationConfig(
                strategy=strategy,
            )
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")

    def create_transformer_config(
        self,
        transformer_dict: typing.Dict[str, typing.Any],
        attention_config: attention_configs.BaseAttentionConfig,
        ffn_config: typing.Optional[ffn_configs.FFNConfig],
        normalization_config: normalization_configs.NormalizationConfig,
        rope_config: typing.Optional[position_configs.RoPEConfig],
        moe_config: typing.Optional[moe_configs.MoEConfig],
    ) -> transformer_configs.TransformerConfig:
        """Create transformer config from parsed components."""
        return transformer_configs.TransformerConfig(
            vocab_size=transformer_dict["vocab_size"],
            block_size=transformer_dict["block_size"],
            hidden_dim=transformer_dict["hidden_dim"],
            n_layers=transformer_dict["n_layers"],
            dropout=transformer_dict["dropout"],
            attention_config=attention_config,
            ffn_config=ffn_config,
            normalization_config=normalization_config,
            rope_config=rope_config,
            moe_config=moe_config,
            bias=transformer_dict["bias"],
        )

    def create_llm_config(
        self,
        token_embedding_config: token_configs.TokenEmbeddingConfig,
        transformer_config: transformer_configs.TransformerConfig,
        output_head_config: head_configs.OutputHeadConfig,
        weight_init_config: weight_init.InitializationConfig,
        position_embedding_config: typing.Optional[
            llm_position_configs.LearnedPositionEmbeddingConfig
        ],
        mtp_config: typing.Optional[head_configs.MultiTokenPredictionConfig],
        architecture: enums.Architecture,
    ) -> typing.Union[llm_configs.GPT2Config, llm_configs.LlamaConfig, llm_configs.DeepSeekConfig]:
        """Create LLM config from parsed components based on architecture."""
        if architecture == enums.Architecture.GPT2:
            if position_embedding_config is None:
                raise ValueError("GPT-2 architecture requires position_embedding_config")
            return llm_configs.GPT2Config(
                token_embedding_config=token_embedding_config,
                position_embedding_config=position_embedding_config,
                transformer_config=transformer_config,
                output_head_config=output_head_config,
                weight_init_config=weight_init_config,
            )
        elif architecture == enums.Architecture.LLAMA3_1:
            return llm_configs.LlamaConfig(
                token_embedding_config=token_embedding_config,
                transformer_config=transformer_config,
                output_head_config=output_head_config,
                weight_init_config=weight_init_config,
            )
        elif architecture == enums.Architecture.DEEPSEEK3:
            if mtp_config is None:
                raise ValueError("DeepSeek architecture requires mtp_config")
            return llm_configs.DeepSeekConfig(
                token_embedding_config=token_embedding_config,
                transformer_config=transformer_config,
                output_head_config=output_head_config,
                mtp_config=mtp_config,
                weight_init_config=weight_init_config,
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture.value}")
