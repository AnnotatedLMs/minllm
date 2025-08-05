# Standard Library
import typing

# Third Party
import torch
from transformers import DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizer


class PaddingCollator(DataCollatorForSeq2Seq):
    """
    Standard padding collator for instruction tuning.

    This is a thin wrapper around HuggingFace's DataCollatorForSeq2Seq
    that ensures proper padding for batched training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: typing.Optional[torch.nn.Module] = None,
        padding: typing.Union[bool, str] = "longest",
        max_length: typing.Optional[int] = None,
        pad_to_multiple_of: typing.Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        """
        Initialize the collator.

        Args:
            tokenizer: Tokenizer to use for padding
            model: Model (used to get pad token if needed)
            padding: Padding strategy ("longest", "max_length", or False)
            max_length: Maximum length to pad to
            pad_to_multiple_of: Pad to multiple of this value
            label_pad_token_id: Token ID to use for padding labels
            return_tensors: Return type ("pt" for PyTorch)
        """
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors,
        )

    def __call__(
        self, features: typing.List[typing.Dict[str, typing.Any]]
    ) -> typing.Dict[str, torch.Tensor]:
        """
        Collate and pad a batch of examples.

        Args:
            features: List of tokenized examples

        Returns:
            Dictionary with padded tensors
        """
        # Use parent class collation
        batch = super().__call__(features)

        # Ensure all expected keys are present
        if "labels" not in batch and "input_ids" in batch:
            # If no labels provided, use input_ids as labels
            batch["labels"] = batch["input_ids"].clone()

        return batch


def create_sft_collator(
    tokenizer: PreTrainedTokenizer,
    packing: bool = False,
    model: typing.Optional[torch.nn.Module] = None,
) -> typing.Union[PaddingCollator, "TensorDataCollatorWithFlattening"]:
    """
    Create the appropriate collator based on configuration.

    Args:
        tokenizer: Tokenizer to use
        packing: Whether to use packing (padding-free) collation
        model: Model (used for padding collator)

    Returns:
        Either PaddingCollator or TensorDataCollatorWithFlattening
    """
    if packing:
        # Third Party
        from posttraining.common.patterns.collation.packing_collator import (
            TensorDataCollatorWithFlattening,
        )

        return TensorDataCollatorWithFlattening()
    else:
        return PaddingCollator(tokenizer=tokenizer, model=model, padding="longest")
