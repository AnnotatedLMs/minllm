"""
Unit tests for LossHandler with logits computation.
"""

# Third Party
import pytest
import torch

# Project
from pretraining.common.models import inputs
from pretraining.common.models import outputs
from pretraining.configs.training import loss_configs
from pretraining.utils.training import loss


class TestLossHandler:
    """Test LossHandler functionality."""

    @pytest.fixture
    def loss_config(self):
        """Create a test loss config."""
        return loss_configs.LossConfig(
            cross_entropy_weight=1.0,
            moe_aux_loss_weight=0.01,
            mtp_loss_weight=0.3,
            z_loss_weight=0.1,
            ignore_index=-100,
        )

    @pytest.fixture
    def loss_handler(self, loss_config):
        """Create a loss handler with test config."""
        return loss.LossHandler(loss_config)

    def test_compute_cross_entropy_loss(self, loss_handler):
        """Test cross-entropy loss computation from logits."""
        batch_size, seq_len, vocab_size = 2, 10, 100

        # Create random logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute loss
        ce_loss = loss_handler.compute_cross_entropy_loss(logits, labels)

        # Check loss properties
        assert isinstance(ce_loss, torch.Tensor)
        assert ce_loss.shape == ()  # Scalar
        assert ce_loss.item() > 0  # Should be positive

    def test_compute_z_loss(self, loss_handler):
        """Test z-loss computation for training stability."""
        batch_size, seq_len, vocab_size = 2, 10, 100

        # Create logits with known values
        logits = torch.ones(batch_size, seq_len, vocab_size) * 2.0

        # Compute z-loss
        z_loss = loss_handler.compute_z_loss(logits)

        # Check z-loss
        assert isinstance(z_loss, torch.Tensor)
        assert z_loss.shape == ()  # Scalar
        # Z-loss should be mean of squared logits = 4.0
        assert torch.isclose(z_loss, torch.tensor(4.0))

    def test_compute_mtp_losses(self, loss_handler):
        """Test multi-token prediction loss computation."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        depth = 3

        # Create MTP logits (list of tensors)
        mtp_logits = [torch.randn(batch_size, seq_len, vocab_size) for _ in range(depth)]

        # Create MTP targets [batch, depth, seq]
        mtp_targets = torch.randint(0, vocab_size, (batch_size, depth, seq_len))

        # Compute MTP losses
        mtp_losses = loss_handler.compute_mtp_losses(mtp_logits, mtp_targets)

        # Check losses
        assert isinstance(mtp_losses, list)
        assert len(mtp_losses) == depth
        for mtp_loss in mtp_losses:
            assert isinstance(mtp_loss, torch.Tensor)
            assert mtp_loss.shape == ()  # Scalar
            assert mtp_loss.item() > 0

    def test_compute_losses_full(self, loss_handler):
        """Test full loss computation from model output."""
        batch_size, seq_len, vocab_size = 2, 10, 100

        # Create model output with single logits (like GPT2/Llama)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        model_output = outputs.ForwardOutput(
            logits=logits,
            mtp_logits=None,
            aux_losses=None,
        )

        # Create training inputs
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        training_inputs = inputs.TrainingInputs(
            input_ids=torch.randint(0, vocab_size, (batch_size, seq_len)),
            labels=labels,
        )

        # Compute losses
        losses = loss_handler.compute_losses(model_output, training_inputs)

        # Check losses
        assert isinstance(losses, dict)
        assert "cross_entropy" in losses
        assert "z_loss" in losses  # Because z_loss_weight > 0
        assert losses["cross_entropy"].shape == ()
        assert losses["z_loss"].shape == ()

    def test_compute_losses_with_mtp_and_aux(self, loss_handler):
        """Test loss computation with MTP and auxiliary losses."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        depth = 3

        # Create model output with all types of outputs (like DeepSeek)
        main_logits = torch.randn(batch_size, seq_len, vocab_size)
        mtp_logits = [torch.randn(batch_size, seq_len, vocab_size) for _ in range(depth)]
        aux_losses = [torch.tensor(0.1), torch.tensor(0.2)]

        model_output = outputs.ForwardOutput(
            logits=main_logits,
            mtp_logits=mtp_logits,
            aux_losses=aux_losses,
        )

        # Create training inputs with MTP targets
        training_inputs = inputs.TrainingInputs(
            input_ids=torch.randint(0, vocab_size, (batch_size, seq_len)),
            labels=torch.randint(0, vocab_size, (batch_size, seq_len)),
            mtp_targets=torch.randint(0, vocab_size, (batch_size, depth, seq_len)),
        )

        # Compute losses
        losses = loss_handler.compute_losses(model_output, training_inputs)

        # Check all loss components
        assert "cross_entropy" in losses
        assert "z_loss" in losses
        assert "aux_loss" in losses
        for i in range(depth):
            assert f"mtp_loss_{i + 1}" in losses

        # Check aux loss aggregation
        expected_aux = sum(aux_losses)
        assert torch.isclose(losses["aux_loss"], expected_aux)

    def test_aggregate_losses(self, loss_handler):
        """Test loss aggregation with configured weights."""
        # Create individual losses
        losses = {
            "cross_entropy": torch.tensor(2.0),
            "z_loss": torch.tensor(1.0),
            "aux_loss": torch.tensor(0.5),
            "mtp_loss_1": torch.tensor(3.0),
            "mtp_loss_2": torch.tensor(2.5),
            "mtp_loss_3": torch.tensor(2.0),
        }

        # Aggregate losses
        total_loss = loss_handler.aggregate_losses(losses)

        # Calculate expected total
        expected = (
            1.0 * 2.0  # cross_entropy
            + 0.1 * 1.0  # z_loss
            + 0.01 * 0.5  # aux_loss
            + 0.3 * ((3.0 + 2.5 + 2.0) / 3)  # averaged MTP
        )

        assert torch.isclose(total_loss, torch.tensor(expected), rtol=1e-5)

    def test_ignore_index_handling(self, loss_handler):
        """Test that ignore_index is properly handled."""
        batch_size, seq_len, vocab_size = 2, 10, 100

        # Create logits and labels with ignore tokens
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[0, 5:] = -100  # Set some tokens to ignore

        # Compute loss
        ce_loss = loss_handler.compute_cross_entropy_loss(logits, labels)

        # Loss should still be valid
        assert isinstance(ce_loss, torch.Tensor)
        assert ce_loss.shape == ()
        assert torch.isfinite(ce_loss)

    def test_format_losses_for_logging(self):
        """Test loss formatting for logging."""
        # Create tensor losses
        losses = {
            "cross_entropy": torch.tensor(2.345),
            "aux_loss": torch.tensor(0.123),
        }

        # Format for logging
        formatted = loss.LossHandler.format_losses_for_logging(losses)

        # Check formatting
        assert isinstance(formatted, dict)
        assert abs(formatted["cross_entropy"] - 2.345) < 1e-6
        assert abs(formatted["aux_loss"] - 0.123) < 1e-6
        assert all(isinstance(v, float) for v in formatted.values())
