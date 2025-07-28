#!/usr/bin/env python3
"""Training script for GPT-2 models.

Usage:
    Single GPU:
        uv run python train_gpt2.py configs/examples/debug/gpt2_debug.yaml

    Multi-GPU with torchrun:
        torchrun --nproc_per_node=4 train_gpt2.py configs/examples/debug/gpt2_debug.yaml
"""

# Standard Library
import argparse
import datetime
import os

# Third Party
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Project
from pretraining.common.patterns.architectures import gpt2
from pretraining.configs import core
from pretraining.configs import loader
from pretraining.configs.model.architectures import gpt
from pretraining.configs.training import execution_configs
from pretraining.data import dataloader
from pretraining.trainer import llm_trainer
from pretraining.utils import logger as logger_utils
from pretraining.utils import torch_utils
from pretraining.utils.training import checkpoint
from pretraining.utils.training import distributed
from pretraining.utils.training import lr_scheduler
from pretraining.utils.training import optimizer


def main(trainer_config: core.TrainerConfig) -> None:
    """Main training function that runs after distributed setup."""

    # Setup device using config
    device = trainer_config.training.execution.setup_device()

    # Setup logging
    logger_utils.setup_logging(rank=distributed.get_global_rank())
    log = logger_utils.get_logger(__name__)

    log.info(f"Using device: {device}")
    log.info(f"Execution strategy: {trainer_config.training.execution.strategy.value}")

    # Set random seed for reproducibility
    torch_utils.seed_all(trainer_config.training.seed)

    # Create model
    log.info("Building GPT-2 model...")
    model = gpt2.GPT2(trainer_config.llm)

    # Move model to device
    model = model.to(device)

    # Wrap model based on execution strategy
    if trainer_config.training.execution.strategy == execution_configs.ExecutionStrategy.DDP:
        log.info("Wrapping model with DDP...")

        dist_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=trainer_config.training.execution.ddp.find_unused_params,
        )

    else:  # SINGLE
        # Use SingleAccelerator for consistent interface
        dist_model = distributed.SingleAccelerator(model)

    # Build optimizer
    optim = optimizer.OptimizerFactory.create_from_config(
        dist_model, trainer_config.training.optimizer, device_type=device.type
    )

    # Build scheduler
    scheduler = lr_scheduler.LRSchedulerFactory.create_from_config(
        optim,
        trainer_config.training.lr_schedule,
        num_training_steps=trainer_config.training.max_iters,
    )

    # Build data loader
    log.info("Setting up data loader...")
    data_loader = dataloader.PretrainDataLoader(
        data_config=trainer_config.training.data,
        batch_config=trainer_config.training.batch,
        device=device,
    )

    # Initialize trainer
    trainer = llm_trainer.LLMTrainer(
        model=model,
        dist_model=dist_model,
        config=trainer_config.training,
        optimizer=optim,
        scheduler=scheduler,
        dataloader=data_loader,
        device=device,
        rank=distributed.get_global_rank(),
        world_size=distributed.get_world_size(),
    )

    # Setup trainer components
    trainer.setup()

    # Setup checkpointer (main process only)
    checkpointer = None
    if distributed.is_main_process():
        checkpointer = checkpoint.Checkpointer(trainer_config.training.checkpoint)

        # Check for existing checkpoint
        checkpoint_data = checkpointer.load_checkpoint(str(device))
        if checkpoint_data is not None:
            log.info("Loading checkpoint...")
            trainer.load_checkpoint_data(checkpoint_data)
            log.info(f"Resumed from iteration {trainer.state.iteration}")

    # Synchronize all processes
    distributed.barrier()

    # Training loop
    try:
        log.info("Starting training...")
        trainer.train()
        log.info("Training complete")

    except KeyboardInterrupt:
        log.info("Training interrupted")

    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        raise

    finally:
        # Save final checkpoint
        if distributed.is_main_process() and checkpointer is not None:
            log.info("Saving final checkpoint...")
            checkpoint_data = trainer.get_checkpoint_data()
            checkpointer.save_checkpoint(checkpoint_data)

            # Save model-only checkpoint for deployment
            checkpointer.save_model_only(
                model.state_dict(),
                {
                    "iteration": trainer.state.iteration,
                    "val_loss": trainer.state.best_val_loss,
                    "config": trainer_config.llm.model_dump(),
                },
            )


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train GPT-2 model")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    # Load configuration
    trainer_config = loader.load_training_config(args.config, gpt.GPT2Config)
    execution_strategy = trainer_config.training.execution.strategy

    # Initialize multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    # Initialize distributed based on strategy
    if execution_strategy == execution_configs.ExecutionStrategy.SINGLE:
        # No distributed setup needed
        main(trainer_config)

    else:  # DDP
        # Setup distributed training
        # torchrun sets these environment variables
        if "RANK" not in os.environ:
            raise RuntimeError(
                f"Execution strategy {execution_strategy.value} requires running with torchrun. "
                "Example: torchrun --nproc_per_node=4 train_gpt2.py config.yaml"
            )

        # Initialize process group
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        if torch.cuda.is_available():
            local_rank = distributed.get_local_rank()
            torch.cuda.set_device(f"cuda:{local_rank}")
            dist.init_process_group(
                backend=backend,
                timeout=datetime.timedelta(minutes=30),
            )
        else:
            dist.init_process_group(
                backend=backend,
                timeout=datetime.timedelta(minutes=30),
            )

        try:
            main(trainer_config)
        finally:
            # Cleanup
            distributed.cleanup_ddp()
