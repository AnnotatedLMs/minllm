"""
Unit tests for data collation.

Tests batch collation for language modeling and multi-token prediction.
Critical for ensuring model receives correctly formatted inputs.
"""

# Third Party
import pytest
import torch
from torch import testing

# Project
from pretraining.data import collator


class TestDataCollator:
    """Test base data collator functionality."""

    @pytest.fixture
    def basic_collator(self) -> collator.DataCollator:
        """Create basic collator."""
        return collator.DataCollator()

    def test_collate_single_batch(self, basic_collator: collator.DataCollator) -> None:
        """Test collating a simple batch."""
        # Create batch with token sequences
        batch = [
            {"input_ids": torch.tensor([1, 2, 3, 4, 5])},
            {"input_ids": torch.tensor([6, 7, 8, 9, 10])},
            {"input_ids": torch.tensor([11, 12, 13, 14, 15])},
        ]

        result = basic_collator(batch)

        # Check output structure
        assert "input_ids" in result
        assert "labels" in result

        # Check shapes
        assert result["input_ids"].shape == (3, 5)
        assert result["labels"].shape == (3, 5)

        # Labels should be a clone of input_ids
        testing.assert_close(result["input_ids"], result["labels"])

    def test_extract_input_ids(self, basic_collator: collator.DataCollator) -> None:
        """Test extracting input_ids from batch."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6])},
        ]

        extracted = basic_collator._extract_input_ids(batch)

        assert len(extracted) == 2
        assert torch.equal(extracted[0], torch.tensor([1, 2, 3]))
        assert torch.equal(extracted[1], torch.tensor([4, 5, 6]))

    def test_stack_tensors(self, basic_collator: collator.DataCollator) -> None:
        """Test stacking tensors into batch."""
        tensors = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
            torch.tensor([7, 8, 9]),
        ]

        stacked = basic_collator._stack_tensors(tensors)

        expected = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        testing.assert_close(stacked, expected)

    def test_create_labels(self, basic_collator: collator.DataCollator) -> None:
        """Test label creation (should be a clone)."""
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        labels = basic_collator._create_labels(input_ids)

        # Should be a clone, not the same tensor
        testing.assert_close(labels, input_ids)
        assert labels is not input_ids  # Different tensor objects


class TestMTPDataCollator:
    """Test multi-token prediction data collator."""

    @pytest.fixture
    def mtp_collator(self) -> collator.MTPDataCollator:
        """Create MTP collator with depth 3."""
        return collator.MTPDataCollator(mtp_depth=3)

    def test_mtp_collate_batch(self, mtp_collator: collator.MTPDataCollator) -> None:
        """Test MTP collation with multiple prediction depths."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])},
            {"input_ids": torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])},
        ]

        result = mtp_collator(batch)

        # Check all outputs are present
        assert "input_ids" in result
        assert "labels" in result
        assert "mtp_targets" in result

        # Check shapes
        assert result["input_ids"].shape == (2, 8)
        assert result["labels"].shape == (2, 8)
        assert result["mtp_targets"].shape == (2, 3, 8)  # batch, depth, seq

    def test_mtp_targets_creation(self, mtp_collator: collator.MTPDataCollator) -> None:
        """Test MTP target creation for different depths."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        mtp_targets = mtp_collator._create_mtp_targets(input_ids)

        # Check shape
        assert mtp_targets.shape == (1, 3, 5)  # batch=1, depth=3, seq=5

        # Check targets for each depth
        # Depth 0: predict 1 token ahead
        expected_depth_0 = torch.tensor([2, 3, 4, 5, -100])
        testing.assert_close(mtp_targets[0, 0], expected_depth_0)

        # Depth 1: predict 2 tokens ahead
        expected_depth_1 = torch.tensor([3, 4, 5, -100, -100])
        testing.assert_close(mtp_targets[0, 1], expected_depth_1)

        # Depth 2: predict 3 tokens ahead
        expected_depth_2 = torch.tensor([4, 5, -100, -100, -100])
        testing.assert_close(mtp_targets[0, 2], expected_depth_2)

    def test_mtp_targets_ignore_index(self, mtp_collator: collator.MTPDataCollator) -> None:
        """Test that MTP targets use -100 for positions without valid targets."""
        input_ids = torch.tensor([[1, 2, 3]])  # Short sequence

        mtp_targets = mtp_collator._create_mtp_targets(input_ids)

        # For depth 2 (predict 3 ahead), all positions should be -100
        # since we can't predict 3 tokens ahead from any position
        assert (mtp_targets[0, 2] == -100).all()

    def test_mtp_depth_configuration(self) -> None:
        """Test MTP collator with different depths."""
        # Test with depth 1
        collator_1 = collator.MTPDataCollator(mtp_depth=1)
        batch = [{"input_ids": torch.tensor([1, 2, 3, 4])}]
        result = collator_1(batch)
        assert result["mtp_targets"].shape == (1, 1, 4)

        # Test with depth 5
        collator_5 = collator.MTPDataCollator(mtp_depth=5)
        result = collator_5(batch)
        assert result["mtp_targets"].shape == (1, 5, 4)


class TestBuildCollator:
    """Test collator factory function."""

    def test_build_standard_collator(self) -> None:
        """Test building standard collator when no MTP config."""

        # Mock config without MTP
        class MockConfig:
            pass

        config = MockConfig()
        built_collator = collator.build_collator(config)

        assert isinstance(built_collator, collator.DataCollator)
        assert not isinstance(built_collator, collator.MTPDataCollator)

    def test_build_mtp_collator(self) -> None:
        """Test building MTP collator when MTP config present."""

        # Mock config with MTP
        class MockMTPConfig:
            n_predict = 4

        class MockConfig:
            mtp = MockMTPConfig()

        config = MockConfig()
        built_collator = collator.build_collator(config)

        assert isinstance(built_collator, collator.MTPDataCollator)
        assert built_collator.mtp_depth == 4
