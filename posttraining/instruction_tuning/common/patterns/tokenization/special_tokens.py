# Standard Library
import typing

# Third Party
import transformers
from transformers import GPTNeoXTokenizerFast
from transformers import LlamaTokenizer
from transformers import LlamaTokenizerFast


def configure_padding_token(
    tokenizer: transformers.PreTrainedTokenizer,
) -> transformers.PreTrainedTokenizer:
    """
    Configure padding token for the tokenizer if not present.

    Args:
        tokenizer: Tokenizer to configure

    Returns:
        Tokenizer with padding token configured
    """
    # Skip if padding token already properly configured
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        return tokenizer

    # Configure based on tokenizer type
    if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
        _add_llama_padding_token(tokenizer)
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        _configure_gpt_neox_tokens(tokenizer)
    elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
        _add_generic_padding_token(tokenizer)

    # Validate configuration
    _validate_special_tokens(tokenizer)

    return tokenizer


def _add_llama_padding_token(tokenizer: typing.Union[LlamaTokenizer, LlamaTokenizerFast]):
    """Add padding token to Llama tokenizers."""
    num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
    assert num_added_tokens in [0, 1], (
        "LlamaTokenizer should only add one special token - the pad_token, "
        "or no tokens if pad token present."
    )


def _configure_gpt_neox_tokens(tokenizer: GPTNeoXTokenizerFast):
    """Configure special tokens for GPT-NeoX tokenizers (used by OLMo)."""
    if tokenizer.bos_token is None:
        # OLMo newer models: use eos_token as bos_token
        tokenizer.bos_token = tokenizer.eos_token
    else:
        # Other models: add padding token
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert num_added_tokens <= 1, (
            "GPTNeoXTokenizer should only add one special token - the pad_token "
            "(or no tokens if already set in SFT)."
        )


def _add_generic_padding_token(tokenizer: transformers.PreTrainedTokenizerFast):
    """Add padding token to generic fast tokenizers."""
    num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
    assert num_added_tokens == 1, (
        "We detected no padding token but add_special_tokens did not add one."
    )


def _validate_special_tokens(tokenizer: transformers.PreTrainedTokenizer):
    """Validate special token configuration."""
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
        "pad token and eos token matching causes issues in our setup."
    )


def validate_model_specific_requirements(
    tokenizer: transformers.PreTrainedTokenizer,
    model_type: str,
    chat_template_name: str,
    add_bos: bool,
    use_fast: bool,
) -> None:
    """
    Validate model-specific tokenizer requirements.

    Args:
        tokenizer: Tokenizer instance
        model_type: Model type from config
        chat_template_name: Name of chat template
        add_bos: Whether to add BOS token
        use_fast: Whether using fast tokenizer

    Raises:
        AssertionError: If requirements not met
    """
    if "olmo" not in model_type:
        return

    # OLMo-specific validations
    if "olmo" in chat_template_name:
        assert not add_bos, "For newer OLMo chat templates, you must *not* run with `--add_bos`."
    else:
        assert add_bos, "For OLMo, you must run with `--add_bos`."

    assert use_fast, "For OLMo, you must use fast tokenizer."

    # Additional check for GPT-NeoX tokenizers with OLMo
    if isinstance(tokenizer, GPTNeoXTokenizerFast) and tokenizer.bos_token is None:
        if "olmo" not in chat_template_name:
            assert add_bos, (
                "For OLMo with GPTNeoX, you must add bos token to the beginning "
                "of the input sequence if using an older chat template."
            )
