"""
Unit tests for checkpoint saving and loading.

Tests checkpoint data structure and state recovery.
Critical for resuming long training runs.
"""

# Standard Library
import tempfile
from pathlib import Path

# Third Party
import pytest
import torch
import torch.nn as nn

# Project
from pretraining.utils.training import state


class SimpleModel(nn.Module):
    """Simple model for testing checkpointing."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestCheckpointData:
    """Test checkpoint data structure and serialization."""

    @pytest.fixture
    def training_state(self) -> state.TrainingState:
        """Create a training state for testing."""
        return state.TrainingState(
            iteration=1000,
            epoch=2,
            tokens_seen=1024000,
            best_val_loss=2.5,
        )

    @pytest.fixture
    def simple_model(self) -> SimpleModel:
        """Create a simple model for testing."""
        return SimpleModel(hidden_dim=128)

    @pytest.fixture
    def optimizer(self, simple_model: SimpleModel) -> torch.optim.Optimizer:
        """Create optimizer for testing."""
        return torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    @pytest.fixture
    def scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        """Create scheduler for testing."""
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)

    def test_model_state_recovery(
        self,
        simple_model: SimpleModel,
    ) -> None:
        """Test recovering model state from checkpoint."""
        # Modify model weights
        with torch.no_grad():
            for param in simple_model.parameters():
                param.data.fill_(1.0)

        # Save original state
        original_state = simple_model.state_dict()

        # Create new model and load checkpoint
        new_model = SimpleModel(hidden_dim=128)

        # Verify weights are different initially
        assert not torch.allclose(new_model.fc1.weight, simple_model.fc1.weight)

        # Load checkpoint into new model
        new_model.load_state_dict(original_state)

        # Verify weights match after loading
        torch.testing.assert_close(new_model.fc1.weight, simple_model.fc1.weight)

    def test_optimizer_state_recovery(
        self,
        simple_model: SimpleModel,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Test recovering optimizer state from checkpoint."""
        # Run a few optimization steps to build optimizer state
        for _ in range(5):
            loss = simple_model(torch.randn(32, 128)).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save optimizer state
        opt_state = optimizer.state_dict()

        # Create new optimizer
        new_optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        # Verify states are different
        assert len(new_optimizer.state) == 0  # No momentum yet
        assert len(optimizer.state) > 0  # Has momentum

        # Load state
        new_optimizer.load_state_dict(opt_state)

        # Verify optimizer state matches
        assert len(new_optimizer.state) == len(optimizer.state)
        assert new_optimizer.state_dict()["param_groups"][0]["lr"] == 1e-3

    def test_training_state_persistence(
        self,
        training_state: state.TrainingState,
    ) -> None:
        """Test that training state is properly persisted."""
        # Modify training state
        training_state.iteration = 2000
        training_state.tokens_seen = 2048000
        training_state.best_val_loss = 2.1

        # Get checkpoint dict
        state_dict = training_state.get_checkpoint_dict()

        # Create new state and load
        new_state = state.TrainingState()
        new_state.load_checkpoint_dict(state_dict)

        # Verify all fields match
        assert new_state.iteration == 2000
        assert new_state.tokens_seen == 2048000
        assert new_state.best_val_loss == 2.1

    def test_training_state_rng_persistence(
        self,
        training_state: state.TrainingState,
    ) -> None:
        """Test that RNG states are properly saved/restored."""
        # Set specific RNG state
        torch.manual_seed(42)

        # Generate a random number
        _ = torch.rand(1).item()

        # Get checkpoint dict (includes RNG state after first random number)
        state_dict = training_state.get_checkpoint_dict()

        # Save the next random number that should be generated
        expected_next_random = torch.rand(1).item()

        # Generate more random numbers to change RNG state
        for _ in range(10):
            torch.rand(1)

        # Verify RNG state has changed
        current_random = torch.rand(1).item()
        assert current_random != expected_next_random

        # Create new state and load (should restore RNG)
        new_state = state.TrainingState()
        new_state.load_checkpoint_dict(state_dict)

        # Should generate the expected next random number
        restored_random_number = torch.rand(1).item()
        assert abs(restored_random_number - expected_next_random) < 1e-6

    def test_checkpoint_dict_structure(
        self,
        simple_model: SimpleModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        training_state: state.TrainingState,
    ) -> None:
        """Test checkpoint dictionary structure for manual save/load."""
        # Create checkpoint dict manually (as trainer would)
        checkpoint_dict = {
            "model": simple_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "training_state": training_state.get_checkpoint_dict(),
            "metadata": {
                "minllm_version": "0.1.0",
                "torch_version": torch.__version__,
            },
        }

        # Save to file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = Path(f.name)
            torch.save(checkpoint_dict, ckpt_path)

        # Load and verify
        loaded_dict = torch.load(ckpt_path, weights_only=False)

        # Check all components present
        assert "model" in loaded_dict
        assert "optimizer" in loaded_dict
        assert "scheduler" in loaded_dict
        assert "training_state" in loaded_dict
        assert "metadata" in loaded_dict

        # Verify training state
        assert loaded_dict["training_state"]["iteration"] == 1000
        assert loaded_dict["training_state"]["tokens_seen"] == 1024000

        # Clean up
        ckpt_path.unlink()

    def test_scheduler_state_recovery(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Test recovering scheduler state from checkpoint."""
        # Step scheduler a few times
        for _ in range(5):
            scheduler.step()

        # Save scheduler state
        scheduler_state = scheduler.state_dict()

        # Create new scheduler
        new_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)

        # Verify states are different
        assert new_scheduler.last_epoch == 0
        assert scheduler.last_epoch == 5

        # Load state
        new_scheduler.load_state_dict(scheduler_state)

        # Verify scheduler state matches
        assert new_scheduler.last_epoch == scheduler.last_epoch

    def test_checkpoint_backward_compatibility(
        self,
        simple_model: SimpleModel,
        training_state: state.TrainingState,
    ) -> None:
        """Test loading checkpoints with missing fields."""
        # Create minimal checkpoint (simulating old version)
        minimal_checkpoint = {
            "model": simple_model.state_dict(),
            "training_state": {
                "iteration": 500,
                "tokens_seen": 512000,
                "epoch": 1,
                # Missing: best_val_loss, rng_state
            },
        }

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = Path(f.name)
            torch.save(minimal_checkpoint, ckpt_path)

        loaded_dict = torch.load(ckpt_path, weights_only=False)

        # Load into new training state (should handle missing fields)
        new_state = state.TrainingState()
        new_state.load_checkpoint_dict(loaded_dict["training_state"])

        # Verify loaded fields
        assert new_state.iteration == 500
        assert new_state.tokens_seen == 512000

        # Missing fields should have defaults
        assert new_state.best_val_loss == float("inf")

        # Clean up
        ckpt_path.unlink()
