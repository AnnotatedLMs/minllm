# Standard Library
import typing
from dataclasses import dataclass

# Third Party
import jaxtyping
import torch
from transformers import DefaultDataCollator


@dataclass
class TensorDataCollatorWithFlattening(DefaultDataCollator):
    """Data collator for padding-free training via sequence packing.

    Core Concept:
    Traditional training pads short sequences to match the longest in a batch,
    wasting computation on meaningless pad tokens. Packing concatenates multiple
    examples into one long sequence, maximizing GPU utilization by ensuring
    every token is meaningful.

    How Packing Works:
    1. Concatenate multiple examples: [Ex1 Ex2 Ex3] → [Ex1Ex2Ex3]
    2. Track boundaries with position_ids that reset at each example
    3. Use attention masks to prevent cross-example attention
    4. Process as single sequence, but examples remain independent

    Benefits:
    - Higher throughput: No wasted computation on padding
    - Better GPU utilization: Especially with varying sequence lengths
    - Memory efficiency: Fewer attention mask operations

    Requirements:
    - Model must support Flash Attention 2 with packing
    - Examples must fit within model's max sequence length
    - Careful handling of position embeddings and attention

    Example:
    Without packing (max_len=512):
    - Example 1: [128 tokens + 384 padding]
    - Example 2: [256 tokens + 256 padding]
    - Example 3: [64 tokens + 448 padding]
    Total: 1536 positions, 448 real tokens, 1088 padding (71% waste!)

    With packing:
    - Packed: [128 + 256 + 64 = 448 tokens]
    Total: 448 positions, 448 real tokens, 0 padding (0% waste!)

    Based on: https://huggingface.co/blog/packing-with-FA2
    """

    return_flash_attn_kwargs: bool = True
    return_position_ids: bool = True
    return_seq_idx: bool = True
    separator_id: int = -100

    def __call__(
        self,
        features: typing.List[typing.Dict[str, jaxtyping.Int[torch.Tensor, "seq"]]],
        return_tensors: typing.Optional[str] = None,
        separator_id: typing.Optional[int] = None,
    ) -> typing.Dict[str, jaxtyping.Int[torch.Tensor, "1 total_seq"]]:
        """
        Collate features by concatenating them.

        Args:
            features: List of tokenized examples
            return_tensors: Return type (uses self.return_tensors if None)
            separator_id: Separator token ID (uses self.separator_id if None)

        Returns:
            Dictionary with concatenated sequences and metadata
        """
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id

        # Initialize tracking variables
        if self.return_flash_attn_kwargs:
            cu_seq_lens = [0]
            max_length = 0
        if self.return_position_ids:
            pos_ids = []
        if self.return_seq_idx:
            seq_idx = []

        # Check if labels are provided
        is_labels_provided = "labels" in features[0]

        # Initialize return dictionary
        ret = {"input_ids": [], "labels": []}

        # Create separator tensor
        separator = torch.tensor(
            [separator_id],
            dtype=features[0]["input_ids"].dtype,
            device=features[0]["input_ids"].device,
        )

        # Process each example
        for s_idx, item in enumerate(features):
            input_ids = item["input_ids"]
            ret["input_ids"].append(input_ids)

            # Handle labels
            if is_labels_provided:
                ret["labels"].append(separator)
                ret["labels"].append(item["labels"][1:])  # Skip first token
            else:
                ret["labels"].append(separator)
                ret["labels"].append(input_ids[1:])  # Use input_ids as labels

            # Update Flash Attention kwargs
            if self.return_flash_attn_kwargs:
                cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids))
                max_length = max(max_length, len(input_ids))

            # Update position IDs
            if self.return_position_ids:
                pos_ids.append(torch.arange(input_ids.numel(), device=input_ids.device))

            # Update sequence indices
            if self.return_seq_idx:
                seq_idx.append(torch.full_like(input_ids, s_idx, dtype=torch.int32))

        # Add Flash Attention metadata
        if self.return_flash_attn_kwargs:
            ret["cu_seq_lens_q"] = ret["cu_seq_lens_k"] = torch.tensor(
                cu_seq_lens, dtype=torch.int32, device=features[0]["input_ids"].device
            )
            ret["max_length_q"] = ret["max_length_k"] = max_length

        # Add position IDs
        if self.return_position_ids:
            ret["position_ids"] = torch.cat(pos_ids, dim=0)[None]

        # Add sequence indices
        if self.return_seq_idx:
            ret["seq_idx"] = torch.cat(seq_idx, dim=0)[None]

        # Concatenate and add batch dimension
        ret["input_ids"] = torch.cat(ret["input_ids"], dim=0)[None]
        ret["labels"] = torch.cat(ret["labels"], dim=0)[None]

        return ret
