# Standard Library
import os
import pathlib
import typing

# Third Party
import numpy as np
import torch


class MemMapDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset backed by memory-mapped token arrays.

    Provides efficient random access to pre-tokenized data stored as numpy arrays.
    Each item is a chunk of contiguous tokens of size `chunk_size`.

    Args:
        paths: Paths to memory-mapped token arrays (.bin files)
        chunk_size: Number of tokens per instance (typically max_sequence_length)
        dtype: Numpy datatype of the arrays (default: np.uint16)
        metadata: Optional metadata to attach to each file
    """

    def __init__(
        self,
        *paths: pathlib.Path,
        chunk_size: int,
        dtype: typing.Type[np.number] = np.uint16,
        metadata: typing.Optional[typing.List[typing.Dict[str, typing.Any]]] = None,
    ):
        self.paths = paths
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.metadata = metadata or [{} for _ in paths]

        self._validate_inputs()
        self._calculate_file_offsets()

    def _validate_inputs(self) -> None:
        """Validate constructor arguments."""
        if not self.paths:
            raise ValueError("At least one path is required")

        if len(self.metadata) != len(self.paths):
            raise ValueError("metadata must have same length as paths")

        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")

    def _calculate_file_offsets(self) -> None:
        """Calculate chunk offsets for each file."""
        self._offsets: typing.List[typing.Tuple[int, int]] = []
        self._file_lengths: typing.List[int] = []
        self._total_chunks = 0

        for path in self.paths:
            num_chunks = self._get_num_chunks_in_file(path)
            self._validate_file_size(path, num_chunks)

            start_offset = self._total_chunks
            end_offset = start_offset + num_chunks

            self._offsets.append((start_offset, end_offset))
            self._file_lengths.append(num_chunks)
            self._total_chunks += num_chunks

    def _get_num_chunks_in_file(self, path: pathlib.Path) -> int:
        """Calculate number of complete chunks in a file."""
        file_size = os.path.getsize(path)
        num_tokens = file_size // self.dtype(0).itemsize
        return num_tokens // self.chunk_size

    def _validate_file_size(self, path: pathlib.Path, num_chunks: int) -> None:
        """Ensure file has at least one complete chunk."""
        if num_chunks == 0:
            file_size = os.path.getsize(path)
            num_tokens = file_size // self.dtype(0).itemsize
            raise ValueError(
                f"File {path} has {num_tokens} tokens, "
                f"which is less than chunk_size={self.chunk_size}"
            )

    def __len__(self) -> int:
        """Total number of chunks across all files."""
        return self._total_chunks

    def __getitem__(self, index: int) -> typing.Dict[str, typing.Any]:
        """
        Get a chunk of tokens.

        Returns:
            Dictionary containing:
            - input_ids: Token tensor of shape (chunk_size,)
            - metadata: Source file metadata
        """
        index = self._normalize_index(index)
        file_idx, local_idx = self._map_index_to_file(index)
        input_ids = self._load_chunk_from_file(file_idx, local_idx)

        return {
            "input_ids": input_ids,
            "metadata": self.metadata[file_idx].copy(),
        }

    def _normalize_index(self, index: int) -> int:
        """Convert negative indices to positive."""
        if index < 0:
            index = len(self) + index

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}")

        return index

    def _map_index_to_file(self, index: int) -> typing.Tuple[int, int]:
        """Map global index to (file_index, local_index) tuple."""
        for file_idx, (start, end) in enumerate(self._offsets):
            if start <= index < end:
                local_idx = index - start
                return file_idx, local_idx

        raise IndexError(f"Could not map index {index} to file")

    def _load_chunk_from_file(self, file_idx: int, local_idx: int) -> torch.Tensor:
        """Load a chunk of tokens from a specific file."""
        path = self.paths[file_idx]
        memmap = self._create_memmap(path)
        tokens = self._extract_tokens(memmap, local_idx)
        return self._tokens_to_tensor(tokens)

    def _create_memmap(self, path: pathlib.Path) -> np.memmap:
        """Create memory-mapped array for efficient access."""
        return np.memmap(path, dtype=self.dtype, mode="r")

    def _extract_tokens(self, memmap: np.memmap, local_idx: int) -> np.ndarray:
        """Extract chunk of tokens at given local index."""
        start_pos = local_idx * self.chunk_size
        end_pos = start_pos + self.chunk_size
        return memmap[start_pos:end_pos]

    def _tokens_to_tensor(self, tokens: np.ndarray) -> torch.Tensor:
        """Convert numpy token array to PyTorch tensor."""
        return torch.from_numpy(tokens.astype(np.int64))
