# Standard Library
import typing

# Third Party
from rich.console import Console
from rich.text import Text
from transformers import PreTrainedTokenizer

# Color palette for visualization
COLORS = ["on red", "on green", "on blue", "on yellow", "on magenta"]


def visualize_token(tokens: typing.List[int], tokenizer: PreTrainedTokenizer):
    """
    Visualize tokens with colors to show token boundaries.

    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer to decode tokens
    """
    console = Console()
    rich_text = Text()

    for i, token in enumerate(tokens):
        color = COLORS[i % len(COLORS)]
        decoded_token = tokenizer.decode(token)
        rich_text.append(f"{decoded_token}", style=color)

    console.print(rich_text)


def visualize_token_role(
    tokens: typing.List[int], masks: typing.List[int], tokenizer: PreTrainedTokenizer
):
    """
    Visualize tokens with colors based on role/mask values.

    Useful for showing which tokens belong to which role in a conversation.

    Args:
        tokens: List of token IDs
        masks: List of mask values (e.g., role IDs)
        tokenizer: Tokenizer to decode tokens
    """
    console = Console()
    rich_text = Text()

    # Process tokens up to the minimum length
    for i in range(min(len(tokens), len(masks))):
        token = tokens[i]
        color = COLORS[masks[i] % len(COLORS)]
        decoded_token = tokenizer.decode(token)
        rich_text.append(f"{decoded_token}", style=color)

    console.print(rich_text)
