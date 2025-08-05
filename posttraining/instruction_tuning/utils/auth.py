# Standard Library
import typing

# Third Party
from huggingface_hub import HfApi


def hf_whoami() -> typing.Dict[str, str]:
    """
    Get the current HuggingFace user information.

    Returns:
        Dictionary with user information including 'name'
    """
    api = HfApi()
    return api.whoami()
