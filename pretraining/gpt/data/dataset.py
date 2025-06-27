"""
Memory-mapped dataset for efficient pretraining data loading.
Expects pre-tokenized data in .bin format.
"""

# Standard Library
import os
import pathlib
import typing

# Third Party
import jaxtyping
import numpy as np
import torch


class PretrainDataset:
    """
    Memory-mapped dataset for GPT pretraining.

    Expects data files in the format:
    - {data_dir}/train.bin: uint16 numpy array of token ids
    - {data_dir}/val.bin: uint16 numpy array of token ids

    The data should be pre-tokenized and saved as numpy arrays.
    """

    def __init__(self, data_dir: pathlib.Path, split: typing.Literal["train", "val"]):
        self.data_dir = data_dir
        self.split = split
        self.data_path = data_dir / f"{split}.bin"

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\nExpected pre-tokenized data in .bin format"
            )

        # Get data length without loading into memory
        self._len = os.path.getsize(self.data_path) // np.dtype(np.uint16).itemsize

    def __len__(self) -> int:
        return self._len

    def get_memmap(self) -> np.memmap:
        """
        Create a new memmap for the data.

        Note: We recreate np.memmap every time to avoid memory leaks.
        See: https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        """
        return np.memmap(self.data_path, dtype=np.uint16, mode="r")

    def _validate_dataset_size(self, data_length: int, block_size: int) -> int:
        """Validate dataset has enough tokens and return max valid position."""
        max_pos = data_length - block_size
        if max_pos <= 0:
            raise ValueError(
                f"Dataset too small ({data_length} tokens) for block_size={block_size}"
            )
        return max_pos

    def _sample_sequence_positions(self, max_pos: int, batch_size: int) -> np.ndarray:
        """Sample random starting positions for sequences in the batch."""
        return np.random.randint(0, max_pos, size=batch_size)

    def _collect_sequences_from_data(
        self, data: np.memmap, positions: np.ndarray, block_size: int
    ) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
        """Collect input and target sequences from the data at given positions."""
        x_list = []
        y_list = []
        for pos in positions:
            # Input: tokens at positions [pos, pos+1, ..., pos+block_size-1]
            x_list.append(data[pos : pos + block_size].astype(np.int64))
            # Target: tokens at positions [pos+1, pos+2, ..., pos+block_size]
            y_list.append(data[pos + 1 : pos + block_size + 1].astype(np.int64))
        return x_list, y_list

    def _create_batch_tensors(
        self, x_list: typing.List[np.ndarray], y_list: typing.List[np.ndarray]
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Stack sequences into batch tensors."""
        x = torch.from_numpy(np.stack(x_list))
        y = torch.from_numpy(np.stack(y_list))
        return x, y

    def sample_batch(
        self, batch_size: int, block_size: int
    ) -> typing.Tuple[
        jaxtyping.Int[torch.Tensor, "batch block_size"],
        jaxtyping.Int[torch.Tensor, "batch block_size"],
    ]:
        """
        Sample a batch of random blocks for language modeling.

        Returns:
            x: Input tokens of shape (batch_size, block_size)
            y: Target tokens of shape (batch_size, block_size)
        """
        data = self.get_memmap()
        max_pos = self._validate_dataset_size(len(data), block_size)
        positions = self._sample_sequence_positions(max_pos, batch_size)
        x_list, y_list = self._collect_sequences_from_data(data, positions, block_size)
        x, y = self._create_batch_tensors(x_list, y_list)
        return x, y
