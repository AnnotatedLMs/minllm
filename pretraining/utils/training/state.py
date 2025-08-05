# Standard Library
import time
import typing

# Third Party
import pydantic
import torch

# Project
from pretraining.configs import base


class TrainingState(base.BaseConfig):
    """Manages training progress and state.

    Tracks iterations, tokens, and timing for LLM pretraining.
    Provides methods to determine when to evaluate, checkpoint, or stop.
    """

    # Override the base config to allow arbitrary types (needed for torch tensors in RNG state)
    model_config = pydantic.ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Progress tracking
    iteration: int = 0
    tokens_seen: int = 0
    epoch: int = 0

    # Timing - use Field with default_factory for time-based defaults
    start_time: float = pydantic.Field(default_factory=time.time)
    last_log_time: float = pydantic.Field(default_factory=time.time)
    last_eval_time: float = pydantic.Field(default_factory=time.time)
    last_checkpoint_time: float = pydantic.Field(default_factory=time.time)

    # Training config (set during init)
    max_iterations: typing.Optional[int] = None
    token_budget: typing.Optional[int] = None
    eval_interval: int = 1000
    checkpoint_interval: int = 5000
    log_interval: int = 10

    # Best metrics tracking
    best_val_loss: float = float("inf")

    def update(self, batch_size: int, seq_length: int) -> None:
        """Update state after training step.

        Args:
            batch_size: Number of sequences in batch
            seq_length: Length of each sequence
        """
        self.iteration += 1
        self.tokens_seen += batch_size * seq_length

    def should_eval(self) -> bool:
        """Check if we should run evaluation."""
        return self.iteration % self.eval_interval == 0

    def should_checkpoint(self) -> bool:
        """Check if we should save checkpoint."""
        return self.iteration % self.checkpoint_interval == 0

    def should_log(self) -> bool:
        """Check if we should log metrics."""
        return self.iteration % self.log_interval == 0

    def should_stop(self) -> bool:
        """Check if training should stop.

        Stops when:
        - Reached token budget (if set)
        - Reached max iterations (if set)
        """
        if self.token_budget and self.tokens_seen >= self.token_budget:
            return True
        if self.max_iterations and self.iteration >= self.max_iterations:
            return True
        return False

    def get_progress(self) -> typing.Dict[str, typing.Any]:
        """Get progress metrics for logging.

        Returns:
            Dictionary with iteration, tokens, time elapsed
        """
        elapsed = time.time() - self.start_time

        return {
            "iteration": self.iteration,
            "tokens_seen": self.tokens_seen,
            "elapsed_time": elapsed,
            "hours": elapsed / 3600,
            "tokens_per_sec": self.tokens_seen / elapsed if elapsed > 0 else 0,
        }

    def get_checkpoint_dict(self) -> typing.Dict[str, typing.Any]:
        """Get state dict for checkpointing.

        Why Save RNG State (for ML researchers):
        - Ensures EXACT reproduction when resuming training
        - Without RNG state: Different random numbers after resume
        - This affects: dropout patterns, data shuffling, augmentations
        - Critical for debugging intermittent training failures

        What Gets Saved:
        - iteration/tokens_seen: Know where training stopped
        - epoch: Current pass through dataset
        - best_val_loss: For model selection
        - torch RNG: Controls CPU operations (dropout, init)
        - cuda RNG: Controls GPU operations (if using GPU)

        Returns:
            Dictionary with all state to save
        """
        return {
            "iteration": self.iteration,
            "tokens_seen": self.tokens_seen,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }

    def load_checkpoint_dict(self, state_dict: typing.Dict[str, typing.Any]) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: Saved state dictionary
        """
        self.iteration = state_dict["iteration"]
        self.tokens_seen = state_dict["tokens_seen"]
        self.epoch = state_dict.get("epoch", 0)
        self.best_val_loss = state_dict.get("best_val_loss", float("inf"))

        # Restore RNG states
        if "rng_state" in state_dict:
            torch.set_rng_state(state_dict["rng_state"]["torch"])
            if torch.cuda.is_available() and state_dict["rng_state"]["cuda"] is not None:
                torch.cuda.set_rng_state(state_dict["rng_state"]["cuda"])

        # Reset timing
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_eval_time = time.time()
        self.last_checkpoint_time = time.time()
