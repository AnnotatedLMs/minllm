"""Unit tests for FineWeb data format support in MemMapDataset."""

# Standard Library
import pathlib
import tempfile
import unittest

# Third Party
import numpy as np
import torch

# Project
from pretraining.data import memmap_dataset


class TestFineWebDataset(unittest.TestCase):
    """Test MemMapDataset with FineWeb/nanoGPT format files."""

    def setUp(self):
        """Create test data files with and without headers."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = pathlib.Path(self.temp_dir.name)

        # Create test tokens
        self.num_tokens = 10000
        self.tokens = np.random.randint(0, 50257, size=self.num_tokens, dtype=np.uint16)

        # Create file with FineWeb header
        self.fineweb_file = self.data_dir / "fineweb_test.bin"
        self._create_fineweb_file(self.fineweb_file, self.tokens)

        # Create raw file without header
        self.raw_file = self.data_dir / "raw_test.bin"
        self.tokens.tofile(self.raw_file)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def _create_fineweb_file(self, path: pathlib.Path, tokens: np.ndarray):
        """Create a FineWeb format file with header."""
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520  # Magic number
        header[1] = 1  # Version
        header[2] = len(tokens)  # Token count

        with open(path, "wb") as f:
            header.tofile(f)
            tokens.tofile(f)

    def test_fineweb_header_detection(self):
        """Test that FineWeb headers are correctly detected."""
        dataset = memmap_dataset.MemMapDataset(self.fineweb_file, chunk_size=1024)

        # Should detect header
        self.assertIsNotNone(dataset._file_headers[0])
        self.assertTrue(dataset._file_headers[0]["has_header"])
        self.assertEqual(dataset._file_headers[0]["num_tokens"], self.num_tokens)

    def test_raw_file_compatibility(self):
        """Test that raw files without headers still work."""
        dataset = memmap_dataset.MemMapDataset(self.raw_file, chunk_size=1024)

        # Should not detect header
        self.assertIsNone(dataset._file_headers[0])

        # Should still load data correctly
        self.assertEqual(len(dataset), self.num_tokens // 1024)

    def test_mixed_formats(self):
        """Test dataset with both FineWeb and raw files."""
        dataset = memmap_dataset.MemMapDataset(self.fineweb_file, self.raw_file, chunk_size=1024)

        # First file has header, second doesn't
        self.assertIsNotNone(dataset._file_headers[0])
        self.assertIsNone(dataset._file_headers[1])

        # Both should contribute chunks
        expected_chunks = (self.num_tokens // 1024) * 2
        self.assertEqual(len(dataset), expected_chunks)

    def test_data_loading_with_header(self):
        """Test that data is correctly loaded from files with headers."""
        dataset = memmap_dataset.MemMapDataset(self.fineweb_file, chunk_size=100)

        # Load first chunk
        item = dataset[0]

        # Check shape and dtype
        self.assertEqual(item["input_ids"].shape, (100,))
        self.assertEqual(item["input_ids"].dtype, torch.int64)

        # Verify we're reading actual token data, not header
        # The first token should not be from the header (20240520 doesn't fit in uint16)
        self.assertLess(item["input_ids"][0].item(), 65536)

    def test_chunk_boundaries_with_header(self):
        """Test that chunks are correctly extracted with header offset."""
        chunk_size = 100
        dataset = memmap_dataset.MemMapDataset(self.fineweb_file, chunk_size=chunk_size)

        # Get consecutive chunks
        chunk1 = dataset[0]["input_ids"]
        chunk2 = dataset[1]["input_ids"]

        # They should be different
        self.assertFalse(torch.equal(chunk1, chunk2))

        # Each should have correct size
        self.assertEqual(len(chunk1), chunk_size)
        self.assertEqual(len(chunk2), chunk_size)


if __name__ == "__main__":
    unittest.main()
