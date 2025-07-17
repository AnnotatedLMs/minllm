# Standard Library
import typing

# Project
# Local
from pretraining.common.base.models import llm
from pretraining.common.patterns.llm import deepseek
from pretraining.common.patterns.llm import gpt2
from pretraining.common.patterns.llm import llama
from pretraining.configs.llm import llm_configs

# TODO: Currently each new llm would require a new import here, eventually could refactor


def create_llm(
    config: typing.Union[
        llm_configs.GPT2Config, llm_configs.LlamaConfig, llm_configs.DeepSeekConfig
    ],
) -> llm.BaseLLM:
    """
    Factory function to create the appropriate LLM based on configuration type.

    This function examines the type of the config object and
    instantiates the corresponding model implementation.

    Args:
        config: Complete LLM configuration object

    Returns:
        Instantiated LLM model ready for training

    Raises:
        ValueError: If configuration type is not supported

    Example:
        >>> from pretraining.configs.parsers.registry import load_training_config
        >>> config = load_training_config("configs/gpt2.yaml")
        >>> model = create_llm(config.llm)
    """
    if isinstance(config, llm_configs.GPT2Config):
        return gpt2.GPT2LLM(config)
    elif isinstance(config, llm_configs.LlamaConfig):
        return llama.LlamaLLM(config)
    elif isinstance(config, llm_configs.DeepSeekConfig):
        return deepseek.DeepSeekLLM(config)
    else:
        raise ValueError(
            f"Unsupported configuration type: {type(config).__name__}. "
            f"Expected one of: GPT2Config, LlamaConfig, DeepSeekConfig"
        )


def get_model_size(
    config: typing.Union[
        llm_configs.GPT2Config, llm_configs.LlamaConfig, llm_configs.DeepSeekConfig
    ],
) -> typing.Dict[str, int]:
    """
    Calculate the approximate model size based on configuration.

    Args:
        config: LLM configuration

    Returns:
        Dictionary with size information:
        - total_params: Total number of parameters
        - embedding_params: Parameters in embeddings
        - transformer_params: Parameters in transformer blocks
        - head_params: Parameters in output head
    """
    vocab_size = config.token_embedding_config.vocab_size
    hidden_dim = config.transformer_config.hidden_dim
    n_layers = config.transformer_config.n_layers

    # Token embeddings
    embedding_params = vocab_size * hidden_dim

    # Position embeddings (if used)
    if config.position_embedding_config is not None:
        embedding_params += config.position_embedding_config.max_position_embeddings * hidden_dim

    # Transformer blocks
    transformer_params = 0
    for _ in range(n_layers):
        # Attention
        attn_config = config.transformer_config.attention_config
        if hasattr(attn_config, "num_kv_heads") and attn_config.num_kv_heads is not None:
            # GQA
            q_params = hidden_dim * hidden_dim
            kv_params = (
                2 * hidden_dim * (hidden_dim * attn_config.num_kv_heads // attn_config.num_heads)
            )
            attn_params = q_params + kv_params + hidden_dim * hidden_dim  # + output proj
        else:
            # Standard MHA
            attn_params = 4 * hidden_dim * hidden_dim  # QKV + output
        transformer_params += attn_params

        # FFN or MoE
        if config.transformer_config.moe_config is not None:
            # MoE (simplified estimate)
            moe_config = config.transformer_config.moe_config
            expert_params = 3 * hidden_dim * hidden_dim  # Simplified
            transformer_params += moe_config.num_experts * expert_params
        else:
            # Standard FFN
            ffn_config = config.transformer_config.ffn_config
            intermediate_dim = ffn_config.intermediate_dim or (4 * hidden_dim)
            if ffn_config.ffn_type == "swiglu":
                transformer_params += 3 * hidden_dim * intermediate_dim  # gate + up + down
            else:
                transformer_params += 2 * hidden_dim * intermediate_dim  # up + down

        # Normalization (2 per block)
        transformer_params += 2 * hidden_dim  # RMSNorm or LayerNorm weights

    # Output head
    head_params = 0
    if not config.output_head_config.tie_word_embeddings:
        head_params = vocab_size * hidden_dim

    # MTP heads if present
    if config.mtp_config is not None:
        mtp_params = config.mtp_config.n_predict * hidden_dim * (hidden_dim + vocab_size)
        head_params += mtp_params

    total_params = embedding_params + transformer_params + head_params

    return {
        "total_params": total_params,
        "embedding_params": embedding_params,
        "transformer_params": transformer_params,
        "head_params": head_params,
    }


def validate_config_architecture_consistency(
    config: typing.Union[
        llm_configs.GPT2Config, llm_configs.LlamaConfig, llm_configs.DeepSeekConfig
    ],
) -> None:
    """
    Validate that the config components match the expected architecture.

    This provides an additional layer of validation beyond what's in
    the config classes themselves.

    Args:
        config: LLM configuration to validate

    Raises:
        ValueError: If configuration is inconsistent with architecture
    """
    if isinstance(config, llm_configs.GPT2Config):
        # GPT-2 specific validations
        if config.position_embedding_config is None:
            raise ValueError("GPT-2 requires learned position embeddings")
        if config.transformer_config.rope_config is not None:
            raise ValueError("GPT-2 should not have RoPE config")
        if config.transformer_config.moe_config is not None:
            raise ValueError("GPT-2 does not use MoE")
        if hasattr(config.transformer_config.attention_config, "num_kv_heads"):
            if (
                config.transformer_config.attention_config.num_kv_heads
                != config.transformer_config.attention_config.num_heads
            ):
                raise ValueError("GPT-2 uses standard multi-head attention, not GQA")

    elif isinstance(config, llm_configs.LlamaConfig):
        # Llama specific validations
        if config.transformer_config.rope_config is None:
            raise ValueError("Llama requires RoPE config")
        if config.transformer_config.moe_config is not None:
            raise ValueError("Standard Llama does not use MoE")
        if not config.output_head_config.tie_word_embeddings:
            raise ValueError("Llama requires tied word embeddings")

    elif isinstance(config, llm_configs.DeepSeekConfig):
        # DeepSeek specific validations
        if config.transformer_config.rope_config is None:
            raise ValueError("DeepSeek requires RoPE config")
        if config.transformer_config.moe_config is None:
            raise ValueError("DeepSeek requires MoE config")
        if not config.output_head_config.tie_word_embeddings:
            raise ValueError("DeepSeek requires tied word embeddings")
