# Standard Library
import typing

# Third Party
import transformers
from transformers import AutoTokenizer

# Project
from posttraining.instruction_tuning.common.patterns.tokenization import chat_templates


def configure_chat_template(
    tokenizer: transformers.PreTrainedTokenizer,
    chat_template_name: str,
    tokenizer_name_or_path: str,
    tokenizer_revision: typing.Optional[str] = None,
    add_bos: bool = False,
) -> transformers.PreTrainedTokenizer:
    """
    Configure chat template for the tokenizer.

    Args:
        tokenizer: Tokenizer to configure
        chat_template_name: Name of chat template to use
        tokenizer_name_or_path: Path to tokenizer (for fallback)
        tokenizer_revision: Tokenizer revision (for fallback)
        add_bos: Whether to prepend BOS token to template

    Returns:
        Tokenizer with chat template configured
    """
    # Set the template
    tokenizer.chat_template = _get_chat_template(
        chat_template_name, tokenizer_name_or_path, tokenizer_revision
    )

    # Add BOS token if requested
    if add_bos:
        tokenizer.chat_template = _add_bos_to_template(tokenizer.chat_template, tokenizer.bos_token)

    return tokenizer


def _get_chat_template(
    template_name: str,
    tokenizer_name_or_path: str,
    tokenizer_revision: typing.Optional[str] = None,
) -> str:
    """
    Get chat template by name or from tokenizer default.

    Args:
        template_name: Name of template to use
        tokenizer_name_or_path: Path to tokenizer
        tokenizer_revision: Tokenizer revision

    Returns:
        Chat template string

    Raises:
        ValueError: If template cannot be found
    """
    # Use predefined template if available
    if template_name in chat_templates.CHAT_TEMPLATES:
        return chat_templates.CHAT_TEMPLATES[template_name]

    # Try to get default template from tokenizer
    try:
        fallback_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, revision=tokenizer_revision
        )
        return fallback_tokenizer.chat_template
    except Exception:
        raise ValueError(
            f"Could not find chat template for {tokenizer_name_or_path}. "
            f"Template '{template_name}' not in predefined templates."
        )


def _add_bos_to_template(
    template: str,
    bos_token: typing.Optional[str],
) -> str:
    """
    Add BOS token to beginning of chat template.

    Args:
        template: Chat template string
        bos_token: BOS token string

    Returns:
        Template with BOS token prepended

    Raises:
        ValueError: If template already contains BOS token
    """
    # Check if template already has BOS token
    if template.startswith("{{ bos_token }}"):
        raise ValueError(
            "You specified add_bos=True, but the chat template already has "
            "a bos_token at the beginning."
        )

    # Check if template starts with literal BOS token
    if bos_token is not None and template.startswith(bos_token):
        raise ValueError(
            "You specified add_bos=True, but the chat template already has "
            "a bos_token at the beginning."
        )

    return "{{ bos_token }}" + template
