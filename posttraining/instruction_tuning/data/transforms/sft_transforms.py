# Standard Library
import copy
import typing

# Third Party
import torch
from transformers import PreTrainedTokenizer

# Dataset keys
INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"
INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
DEFAULT_SFT_MESSAGES_KEY = "messages"

# Special token for masking
IGNORE_INDEX = -100


def sft_tulu_tokenize_and_truncate_v1(
    row: typing.Dict[str, typing.Any], tokenizer: PreTrainedTokenizer, max_seq_length: int
) -> typing.Dict[str, typing.Any]:
    """Tokenize chat messages with selective masking for instruction tuning.

    Core Operation:
    This function implements the key insight of instruction tuning: the model
    should learn to generate responses, not memorize the conversation format.
    It masks all non-assistant tokens with IGNORE_INDEX (-100) so they don't
    contribute to the loss.

    Masking Strategy:
    1. System message → Masked (context only)
    2. User message → Masked (instruction only)
    3. Assistant message → Not masked (learn to generate)

    Example:
    Input messages:
    - {"role": "system", "content": "You are helpful"}
    - {"role": "user", "content": "What is 2+2?"}
    - {"role": "assistant", "content": "2+2 equals 4"}

    After tokenization and masking:
    - Tokens: [system_tokens, user_tokens, assistant_tokens]
    - Labels: [-100, -100, ..., -100, -100, ..., actual_token_ids]

    This ensures the model learns P(response|instruction) rather than
    P(entire_conversation), making it better at following new instructions.

    Args:
        row: Dataset row with 'messages' field
        tokenizer: Tokenizer with chat template
        max_seq_length: Maximum sequence length

    Returns:
        Dictionary with input_ids, labels (masked), and attention_mask
    """
    messages = row["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    # Tokenize the full conversation
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )

    # Create labels by copying input_ids
    labels = input_ids.clone()

    # Mask the non-assistant parts to avoid loss computation
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # Calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]

            # Calculate the end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # For messages followed by assistant, add generation prompt
                # to avoid including the assistant prefix in the loss
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # For last message or non-assistant-followed messages
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]

            # Set labels to IGNORE_INDEX for non-assistant parts
            labels[:, message_start_idx:message_end_idx] = IGNORE_INDEX

            # Stop if we've reached the max length
            if max_seq_length and message_end_idx >= max_seq_length:
                break

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    # Flatten tensors and return
    row[INPUT_IDS_KEY] = input_ids.flatten()
    row[LABELS_KEY] = labels.flatten()
    row[ATTENTION_MASK_KEY] = attention_mask.flatten()

    return row


def sft_tulu_filter_v1(row: typing.Dict[str, typing.Any], tokenizer: PreTrainedTokenizer) -> bool:
    """
    Filter out examples that have no trainable tokens.

    Args:
        row: Tokenized dataset row
        tokenizer: Tokenizer (not used but kept for consistency)

    Returns:
        True if the example has at least one non-masked token
    """
    return any(x != IGNORE_INDEX for x in row[LABELS_KEY])


def sft_tokenize_v1(
    row: typing.Dict[str, typing.Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
) -> typing.Dict[str, typing.Any]:
    """
    Simple tokenization without masking - train on all tokens.

    Args:
        row: Dataset row with messages
        tokenizer: Tokenizer with chat template
        sft_messages_key: Key containing messages

    Returns:
        Dictionary with tokenized inputs
    """
    # Extract prompt (all messages except last)
    if len(row[sft_messages_key]) == 1:
        prompt = row[sft_messages_key]
    else:
        prompt = row[sft_messages_key][:-1]

    # Tokenize prompt and full conversation
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])

    # Copy input_ids as labels (no masking)
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    row[LABELS_KEY] = labels

    return row


def sft_tokenize_mask_out_prompt_v1(
    row: typing.Dict[str, typing.Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
) -> typing.Dict[str, typing.Any]:
    """
    Tokenize and mask out the prompt tokens in labels.

    This simpler approach masks everything except the final response.

    Args:
        row: Dataset row with messages
        tokenizer: Tokenizer with chat template
        sft_messages_key: Key containing messages

    Returns:
        Dictionary with tokenized inputs and masked labels
    """
    # Extract prompt (all messages except last)
    if len(row[sft_messages_key]) == 1:
        prompt = row[sft_messages_key]
    else:
        prompt = row[sft_messages_key][:-1]

    # Tokenize prompt and full conversation
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])

    # Create labels with prompt masked out
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    labels[: len(row[INPUT_IDS_PROMPT_KEY])] = [IGNORE_INDEX] * len(row[INPUT_IDS_PROMPT_KEY])
    row[LABELS_KEY] = labels

    return row


def sft_filter_v1(
    row: typing.Dict[str, typing.Any],
    tokenizer: PreTrainedTokenizer,
    max_prompt_token_length: typing.Optional[int] = None,
    max_token_length: typing.Optional[int] = None,
    need_contain_labels: bool = True,
) -> bool:
    """
    Filter examples based on length constraints.

    Args:
        row: Tokenized dataset row
        tokenizer: Tokenizer (not used but kept for consistency)
        max_prompt_token_length: Maximum prompt length
        max_token_length: Maximum total sequence length
        need_contain_labels: Whether to require trainable tokens

    Returns:
        True if the example passes all filters
    """
    # Check prompt length
    max_prompt_token_length_ok = True
    if max_prompt_token_length is not None:
        max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= max_prompt_token_length

    # Check total length
    max_token_length_ok = True
    if max_token_length is not None:
        max_token_length_ok = len(row[INPUT_IDS_KEY]) <= max_token_length

    # Check for trainable tokens
    contain_some_labels = any(x != IGNORE_INDEX for x in row[LABELS_KEY])

    return (
        max_prompt_token_length_ok
        and max_token_length_ok
        and (contain_some_labels or not need_contain_labels)
    )
