"""
Unit tests for memory-mapped dataset.

Tests MemMapDataset for proper chunking, boundary handling, and iteration.
Critical for ensuring training data is correctly loaded and chunked.
"""

# Standard Library
import pathlib
import tempfile

# Third Party
import numpy as np
import pytest
import torch
from torch import testing

# Project
from pretraining.data import memmap_dataset


class TestMemMapDataset:
    """Test memory-mapped dataset functionality."""

    @pytest.fixture
    def sample_data_file(self) -> pathlib.Path:
        """Create a temporary data file with known content."""
        # Create sample token data
        num_tokens = 1000
        tokens = np.arange(num_tokens, dtype=np.uint16)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            tokens.tofile(f)
            return pathlib.Path(f.name)

    def test_dataset_initialization(self, sample_data_file: pathlib.Path) -> None:
        """Test dataset initializes correctly."""
        chunk_size = 128
        dataset = memmap_dataset.MemMapDataset(
            sample_data_file,
            chunk_size=chunk_size,
        )

        # Check basic properties
        assert dataset.chunk_size == chunk_size
        assert dataset.dtype == np.uint16
        assert len(dataset) == 1000 // chunk_size  # Should be 7 full chunks

        # Clean up
        sample_data_file.unlink()

    def test_chunk_boundaries(self, sample_data_file: pathlib.Path) -> None:
        """Test that chunks respect sequence boundaries."""
        chunk_size = 128
        dataset = memmap_dataset.MemMapDataset(
            sample_data_file,
            chunk_size=chunk_size,
        )

        # Get first chunk
        first_chunk = dataset[0]
        assert isinstance(first_chunk, dict)
        assert "input_ids" in first_chunk
        assert "metadata" in first_chunk
        assert len(first_chunk["input_ids"]) == chunk_size
        assert first_chunk["input_ids"][0] == 0
        assert first_chunk["input_ids"][-1] == 127

        # Get second chunk
        second_chunk = dataset[1]
        assert second_chunk["input_ids"][0] == 128
        assert second_chunk["input_ids"][-1] == 255

        # Verify no overlap
        assert first_chunk["input_ids"][-1] + 1 == second_chunk["input_ids"][0]

        # Clean up
        sample_data_file.unlink()

    def test_last_chunk_handling(self, sample_data_file: pathlib.Path) -> None:
        """Test handling of incomplete last chunk."""
        chunk_size = 128
        dataset = memmap_dataset.MemMapDataset(
            sample_data_file,
            chunk_size=chunk_size,
        )

        # 1000 tokens / 128 chunk = 7 full chunks + 104 remaining
        # The dataset should only return full chunks
        assert len(dataset) == 7

        # Last accessible chunk should be complete
        last_chunk = dataset[-1]
        assert len(last_chunk["input_ids"]) == chunk_size
        assert last_chunk["input_ids"][0] == 768  # 6 * 128
        assert last_chunk["input_ids"][-1] == 895  # 768 + 127

        # Clean up
        sample_data_file.unlink()

    def test_out_of_bounds_access(self, sample_data_file: pathlib.Path) -> None:
        """Test that out-of-bounds access raises appropriate error."""
        chunk_size = 128
        dataset = memmap_dataset.MemMapDataset(
            sample_data_file,
            chunk_size=chunk_size,
        )

        # Should raise IndexError
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]

        with pytest.raises(IndexError):
            _ = dataset[-len(dataset) - 1]

        # Clean up
        sample_data_file.unlink()

    def test_empty_file_handling(self) -> None:
        """Test that empty data file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            empty_file = pathlib.Path(f.name)

        # Should raise ValueError for empty file
        with pytest.raises(ValueError, match="has 0 tokens"):
            _ = memmap_dataset.MemMapDataset(
                empty_file,
                chunk_size=128,
            )

        # Clean up
        empty_file.unlink()

    def test_chunk_type_conversion(self, sample_data_file: pathlib.Path) -> None:
        """Test that chunks are properly converted to tensors."""
        chunk_size = 128
        dataset = memmap_dataset.MemMapDataset(
            sample_data_file,
            chunk_size=chunk_size,
        )

        chunk = dataset[0]

        # Should be a dict with tensor
        assert isinstance(chunk, dict)
        assert isinstance(chunk["input_ids"], torch.Tensor)
        assert chunk["input_ids"].dtype == torch.int64

        # Clean up
        sample_data_file.unlink()

    def test_reproducible_access(self, sample_data_file: pathlib.Path) -> None:
        """Test that repeated access returns same data."""
        chunk_size = 128
        dataset = memmap_dataset.MemMapDataset(
            sample_data_file,
            chunk_size=chunk_size,
        )

        # Access same index multiple times
        chunk1 = dataset[3]
        chunk2 = dataset[3]

        # Should be identical
        testing.assert_close(chunk1["input_ids"], chunk2["input_ids"])

        # Clean up
        sample_data_file.unlink()

    def test_large_chunk_size(self, sample_data_file: pathlib.Path) -> None:
        """Test behavior with chunk size larger than file."""
        chunk_size = 2000  # Larger than 1000 tokens in file

        # Should raise ValueError for chunk size larger than file
        with pytest.raises(ValueError, match="has 1000 tokens"):
            _ = memmap_dataset.MemMapDataset(
                sample_data_file,
                chunk_size=chunk_size,
            )

        # Clean up
        sample_data_file.unlink()
