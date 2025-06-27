# Standard Library
import pathlib
import typing

# Third Party
import torch
import torch.nn as nn

# Project
from pretraining.gpt.utils import config


class Checkpointer:
    """
    Handles saving and loading training checkpoints.

    Checkpoint format:
    {
        'model': model_state_dict,
        'optimizer': optimizer_state_dict,
        'model_args': dict of model constructor args,
        'iter_num': current iteration,
        'best_val_loss': best validation loss seen,
        'config': full training config
    }
    """

    def __init__(self, out_dir: str, config: config.TrainerConfig) -> None:
        self.out_dir = pathlib.Path(out_dir)
        self.config = config
        self.checkpoint_path = self.out_dir / "ckpt.pt"

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iter_num: int,
        best_val_loss: float,
        model_args: typing.Dict[str, typing.Any],
        config: typing.Dict[str, typing.Any],
    ) -> None:
        """
        Save a training checkpoint.

        Args:
            model: Model to save (will handle DDP unwrapping)
            optimizer: Optimizer state to save
            iter_num: Current iteration number
            best_val_loss: Best validation loss achieved
            model_args: Model constructor arguments
            config: Full configuration dict
        """
        # Unwrap DDP if needed
        raw_model = model.module if hasattr(model, "module") else model

        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": model_args,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "config": config,
        }

        print(f"Saving checkpoint to {self.checkpoint_path}")

        # Save to temporary file first, then rename (atomic operation)
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.rename(self.checkpoint_path)

    def load_checkpoint(self, device: str) -> typing.Dict[str, typing.Any]:
        """
        Load a training checkpoint.

        Args:
            device: Device to load tensors to

        Returns:
            Dictionary containing checkpoint data
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path}")

        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=device)

        return checkpoint

    def checkpoint_exists(self) -> bool:
        """Check if a checkpoint exists."""
        return self.checkpoint_path.exists()

    def save_final_model(self, model: nn.Module, iter_num: int, val_loss: float) -> None:
        """
        Save just the final model weights (no optimizer state).
        Useful for deployment/inference.
        """
        # Unwrap DDP if needed
        raw_model = model.module if hasattr(model, "module") else model

        final_path = self.out_dir / f"model_iter{iter_num}_loss{val_loss:.4f}.pt"

        model_state = {
            "model": raw_model.state_dict(),
            "config": raw_model.config,
            "iter_num": iter_num,
            "val_loss": val_loss,
        }

        print(f"Saving final model to {final_path}")
        torch.save(model_state, final_path)
