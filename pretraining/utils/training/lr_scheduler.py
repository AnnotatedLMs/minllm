# Standard Library
import math
import typing

# Third Party
import torch
import transformers
from torch.optim import lr_scheduler

# Project
from pretraining.configs.training import lr_configs


class LRSchedulerFactory:
    """Factory for creating learning rate schedulers used in LLM training.

    Model Effects:
    - Controls learning dynamics throughout training
    - Enables stable convergence during warmup
    - Prevents overfitting in later training stages

    Core Operations:
    - Creates schedulers that modify learning rate over time
    - Supports warmup, decay, and multi-phase schedules
    - Returns PyTorch scheduler objects for training loops

    Training Operations (what LR schedulers do):
    1. Track current training step/epoch
    2. Calculate learning rate multiplier based on schedule
    3. Update optimizer's learning rate before each step
    4. Warmup: Gradually increase LR from 0 to target
    5. Decay: Gradually decrease LR to approach minimum
    6. Apply multiplier to all parameter groups in optimizer
    """

    @staticmethod
    def create_constant(
        optimizer: torch.optim.Optimizer,
    ) -> lr_scheduler.LambdaLR:
        """
        Create constant learning rate scheduler (no decay).

        Keeps learning rate constant throughout training.

        Used by:
        - GPT-2 (nanoGPT): When decay_lr=False in config

        Args:
            optimizer: Optimizer to schedule

        Returns:
            LambdaLR scheduler that maintains constant LR
        """
        return lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    @staticmethod
    def create_cosine_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> lr_scheduler.LambdaLR:
        """
        Create cosine scheduler with linear warmup.

        Most common schedule for LLM training:
        - Linear warmup from 0 to initial LR over num_warmup_steps
        - Cosine decay to ~0 over remaining steps
        - num_cycles=0.5 gives half-cosine (most common)

        Used by:
        - GPT-2 (nanoGPT): Implements this manually with warmup=2000, total=600000
        - Llama 1/2: Uses this schedule with warmup=2000 per paper
        - Llama 3: Uses this with warmup=8000, total=1.2M steps
          https://arxiv.org/pdf/2407.21783

        Uses transformers library implementation.

        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total training steps
            num_cycles: Number of cosine cycles (0.5 = half cosine)
            last_epoch: Last epoch number

        Returns:
            LambdaLR scheduler from transformers
        """
        return transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )

    @staticmethod
    def create_multiphase_schedule(
        optimizer: torch.optim.Optimizer,
        phase_steps: typing.List[int],
        phase_lr_funcs: typing.List[typing.Callable[[int], float]],
    ) -> lr_scheduler.LambdaLR:
        """
        Create a multi-phase learning rate scheduler.

        Allows different LR schedules for different training phases.
        Each phase has a step count and a function that computes LR multiplier.

        Used by:
        - DeepSeek-V3: Complex 3-phase schedule (warmup, constant, cosine decay)

        Args:
            optimizer: Optimizer to schedule
            phase_steps: List of steps for each phase [phase1_steps, phase2_steps, ...]
            phase_lr_funcs: List of functions that take (step_in_phase) -> lr_multiplier

        Returns:
            LambdaLR scheduler with multi-phase schedule

        Example:
            >>> # DeepSeek-V3 style: warmup 2k, constant 10T tokens, cosine decay 4.3T tokens
            >>> # Assuming 1M tokens per step
            >>> phase_steps = [2000, 10_000_000, 4_300_000]
            >>> phase_lr_funcs = [
            ...     lambda s: s / 2000,  # Linear warmup
            ...     lambda s: 1.0,       # Constant
            ...     lambda s: 0.5 * (1 + math.cos(math.pi * s / 4_300_000)) * 0.9 + 0.1  # Cosine to 0.1x
            ... ]
        """
        cumulative_steps = [0]
        for steps in phase_steps:
            cumulative_steps.append(cumulative_steps[-1] + steps)

        def lr_lambda(current_step: int) -> float:
            # Find which phase we're in
            for i, (start, end) in enumerate(zip(cumulative_steps[:-1], cumulative_steps[1:])):
                if start <= current_step < end:
                    step_in_phase = current_step - start
                    return phase_lr_funcs[i](step_in_phase)
            # After all phases, use last phase's final value
            return phase_lr_funcs[-1](phase_steps[-1] - 1)

        return lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def create_from_config(
        optimizer: torch.optim.Optimizer,
        config: lr_configs.LearningRateScheduleConfig,
        num_training_steps: int,
    ) -> lr_scheduler.LRScheduler:
        """
        Create learning rate scheduler from configuration object.

        Args:
            optimizer: Optimizer to schedule
            config: LR schedule configuration
            num_training_steps: Total number of training steps

        Returns:
            Configured LR scheduler
        """
        if config.schedule_type == "constant":
            return LRSchedulerFactory.create_constant(optimizer)

        elif config.schedule_type in ["cosine", "cosine_with_warmup"]:
            # Default num_cycles to 0.5 if not specified
            num_cycles = config.num_cycles if config.num_cycles is not None else 0.5

            return LRSchedulerFactory.create_cosine_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.warmup_iters,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            )

        elif config.schedule_type == "multiphase":
            if config.phase_steps is None or config.phase_names is None:
                raise ValueError("multiphase schedule requires phase_steps and phase_names")

            # Create phase functions based on phase names
            phase_lr_funcs = []
            for i, phase_name in enumerate(config.phase_names):
                if phase_name == "warmup":
                    # Linear warmup
                    warmup_steps = config.phase_steps[i]
                    phase_lr_funcs.append(lambda s, steps=warmup_steps: s / steps)

                elif phase_name == "constant":
                    # Constant at full LR
                    phase_lr_funcs.append(lambda s: 1.0)

                elif phase_name == "decay":
                    # Cosine decay to min_lr
                    decay_steps = config.phase_steps[i]
                    min_lr_ratio = config.min_lr / optimizer.param_groups[0]["lr"]
                    phase_lr_funcs.append(
                        lambda s, steps=decay_steps, min_ratio=min_lr_ratio: 0.5
                        * (1 + math.cos(math.pi * s / steps))
                        * (1 - min_ratio)
                        + min_ratio
                    )

                else:
                    raise ValueError(f"Unknown phase name: {phase_name}")

            return LRSchedulerFactory.create_multiphase_schedule(
                optimizer=optimizer,
                phase_steps=config.phase_steps,
                phase_lr_funcs=phase_lr_funcs,
            )

        else:
            raise ValueError(f"Unknown schedule type: {config.schedule_type}")
