# Standard Library
import typing

# Third Party
import torch
from transformers import get_scheduler as get_hf_scheduler


class SchedulerFactory:
    """Factory for creating learning rate schedulers for SFT.

    Provides both HuggingFace and PyTorch scheduler options with
    configurations optimized for instruction tuning.
    """

    @staticmethod
    def create_cosine_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create cosine schedule with linear warmup.

        The most common schedule for fine-tuning:
        1. Linear warmup from 0 to learning_rate over warmup steps
        2. Cosine decay from learning_rate to 0

        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total training steps
            num_cycles: Number of cosine cycles (0.5 = half cosine)
            last_epoch: Last epoch number (-1 to start fresh)

        Returns:
            Configured scheduler
        """
        return get_hf_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )

    @staticmethod
    def create_linear_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create linear schedule with warmup.

        Simple linear decay after warmup:
        1. Linear warmup from 0 to learning_rate
        2. Linear decay from learning_rate to 0

        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total training steps
            last_epoch: Last epoch number

        Returns:
            Configured scheduler
        """
        return get_hf_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

    @staticmethod
    def create_constant_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        last_epoch: int = -1,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create constant schedule with warmup.

        Maintains constant learning rate after warmup:
        1. Linear warmup from 0 to learning_rate
        2. Constant at learning_rate

        Good for short fine-tuning runs where decay might be harmful.

        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            last_epoch: Last epoch number

        Returns:
            Configured scheduler
        """
        return get_hf_scheduler(
            name="constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            last_epoch=last_epoch,
        )

    @staticmethod
    def create_polynomial_decay(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        lr_end: float = 1e-7,
        power: float = 1.0,
        last_epoch: int = -1,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create polynomial decay schedule with warmup.

        Polynomial decay is more flexible than linear:
        - power=1.0: equivalent to linear
        - power>1.0: slower initial decay, faster final decay
        - power<1.0: faster initial decay, slower final decay

        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total training steps
            lr_end: Final learning rate
            power: Polynomial power
            last_epoch: Last epoch number

        Returns:
            Configured scheduler
        """
        return get_hf_scheduler(
            name="polynomial",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_end=lr_end,
            power=power,
            last_epoch=last_epoch,
        )

    @staticmethod
    def create_from_name(
        name: str,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: typing.Optional[int] = None,
        **kwargs,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create scheduler by name with common SFT defaults.

        Args:
            name: Scheduler name (cosine, linear, constant, polynomial)
            optimizer: Optimizer to schedule
            num_warmup_steps: Warmup steps
            num_training_steps: Total steps (not needed for constant)
            **kwargs: Additional scheduler-specific arguments

        Returns:
            Configured scheduler

        Raises:
            ValueError: If scheduler name not recognized
        """
        if name == "cosine":
            return SchedulerFactory.create_cosine_with_warmup(
                optimizer, num_warmup_steps, num_training_steps, **kwargs
            )
        elif name == "linear":
            return SchedulerFactory.create_linear_with_warmup(
                optimizer, num_warmup_steps, num_training_steps, **kwargs
            )
        elif name == "constant":
            return SchedulerFactory.create_constant_with_warmup(
                optimizer, num_warmup_steps, **kwargs
            )
        elif name == "polynomial":
            return SchedulerFactory.create_polynomial_decay(
                optimizer, num_warmup_steps, num_training_steps, **kwargs
            )
        else:
            # Fall back to HuggingFace scheduler
            return get_hf_scheduler(
                name=name,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **kwargs,
            )
