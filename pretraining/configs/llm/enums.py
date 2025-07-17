# Standard Library
import enum


class Architecture(enum.Enum):
    """Supported model architectures."""

    GPT2 = "gpt2"
    LLAMA3_1 = "llama3.1"  # Match the YAML file
    DEEPSEEK3 = "deepseek3"
