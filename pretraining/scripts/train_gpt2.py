#!/usr/bin/env python3
"""
Training script for GPT-2 models.

Entry point flow:
1. Parse command line arguments for config path
2. Load training configuration with GPT-2 specific model config
3. Initialize multiprocessing with spawn method
4. Branch based on execution strategy (SINGLE vs DDP)
5. For DDP: verify torchrun environment, initialize process group
6. Run main training function
7. Cleanup distributed resources if using DDP

Main training flow:
1. Setup device based on execution strategy
2. Initialize logging for current rank
3. Set random seeds for reproducibility
4. Build GPT-2 model from config
5. Wrap model (DDP for multi-GPU, SingleAccelerator for single)
6. Create optimizer and learning rate scheduler
7. Load checkpoint if resuming (main process only)
8. Build data pipeline: dataset → iterable wrapper → collator → dataloader
9. Create loss handler and evaluator
10. Initialize trainer with all components
11. Run training loop with periodic evaluation
12. Save final checkpoint and model-only checkpoint

Usage:
    Single GPU:
        uv run python train_gpt2.py configs/examples/debug/gpt2_debug.yaml

    Multi-GPU with torchrun:
        torchrun --nproc_per_node=4 train_gpt2.py configs/examples/debug/gpt2_debug.yaml
"""

# Standard Library
import argparse
import os

# Third Party
import torch

# Project
from pretraining.common.patterns.architectures import gpt2
from pretraining.configs import core
from pretraining.configs import loader
from pretraining.configs.model.architectures import gpt
from pretraining.configs.training import execution_configs
from pretraining.data import builder
from pretraining.data import collator
from pretraining.trainer import llm_trainer
from pretraining.utils import logger as logger_utils
from pretraining.utils import torch_utils
from pretraining.utils.training import checkpoint
from pretraining.utils.training import distributed
from pretraining.utils.training import evaluation
from pretraining.utils.training import launcher
from pretraining.utils.training import loss
from pretraining.utils.training import lr_scheduler
from pretraining.utils.training import optimizer


def main(trainer_config: core.TrainerConfig) -> None:
    """Main training function that runs after distributed setup."""

    device = trainer_config.training.execution.setup_device()

    logger_utils.setup_logging(rank=distributed.get_global_rank())
    log = logger_utils.get_logger(__name__)

    log.info(f"Using device: {device}")
    log.info(f"Execution strategy: {trainer_config.training.execution.strategy.value}")

    torch_utils.seed_all(trainer_config.training.seed)

    log.info("Building GPT-2 model...")
    model = gpt2.GPT2(trainer_config.llm)

    model = model.to(device)

    if trainer_config.training.execution.strategy == execution_configs.ExecutionStrategy.DDP:
        log.info("Wrapping model with DDP...")

        dist_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=trainer_config.training.execution.ddp.find_unused_params,
        )

    else:  # SINGLE
        dist_model = distributed.SingleAccelerator(model)

    optim = optimizer.OptimizerFactory.create_from_config(
        dist_model, trainer_config.training.optimizer, device_type=device.type
    )

    scheduler = lr_scheduler.LRSchedulerFactory.create_from_config(
        optim,
        trainer_config.training.lr_schedule,
        num_training_steps=trainer_config.training.max_iters,
    )

    start_index = 0
    epoch = 0
    checkpoint_data = None

    checkpointer = None
    if distributed.is_main_process():
        checkpointer = checkpoint.Checkpointer(trainer_config.training.checkpoint)
        checkpoint_data = checkpointer.load_checkpoint(str(device))
        if checkpoint_data is not None:
            log.info("Found checkpoint, will resume training...")
            if hasattr(checkpoint_data, "training_state_dict"):
                start_index = checkpoint_data.training_state_dict.get("tokens_seen", 0)
                epoch = checkpoint_data.training_state_dict.get("epoch", 0)

    distributed.barrier()

    log.info("Setting up data pipeline...")

    train_dataset = builder.build_memmap_dataset(
        config=trainer_config.training.data,
        split="train",
        chunk_size=trainer_config.training.batch.sequence_length,
    )

    train_dataset = builder.build_iterable_dataset(
        base_dataset=train_dataset,
        training_config=trainer_config.training,
        start_index=start_index,
        epoch=epoch,
        drop_last=True,
    )

    data_collator = collator.DataCollator()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=trainer_config.training.batch.batch_size,
        collate_fn=data_collator,
        num_workers=trainer_config.training.data.num_workers,
        pin_memory=trainer_config.training.data.pin_memory,
        drop_last=True,
        persistent_workers=trainer_config.training.data.num_workers > 0,
    )

    log.info("Setting up validation data...")

    val_dataset = builder.build_memmap_dataset(
        config=trainer_config.training.data,
        split="val",
        chunk_size=trainer_config.training.batch.sequence_length,
    )

    val_sampler = builder.build_distributed_sampler(
        dataset=val_dataset,
        shuffle=False,
        drop_last=False,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=trainer_config.training.batch.batch_size,
        sampler=val_sampler,
        collate_fn=data_collator,
        num_workers=trainer_config.training.data.num_workers,
        pin_memory=trainer_config.training.data.pin_memory,
        drop_last=False,
    )

    loss_handler = loss.LossHandler(trainer_config.training.loss)

    evaluator = evaluation.Evaluator(
        loss_handler=loss_handler,
        num_eval_batches=trainer_config.training.evaluation.num_eval_batches,
    )

    trainer = llm_trainer.LLMTrainer(
        model=model,
        dist_model=dist_model,
        config=trainer_config.training,
        optimizer=optim,
        scheduler=scheduler,
        loss_handler=loss_handler,
        evaluator=evaluator,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        rank=distributed.get_global_rank(),
        world_size=distributed.get_world_size(),
    )

    trainer.setup()

    if checkpoint_data is not None and distributed.is_main_process():
        log.info("Loading checkpoint...")
        trainer.load_checkpoint_data(checkpoint_data)
        log.info(f"Resumed from iteration {trainer.state.iteration}")

    distributed.barrier()

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
        if distributed.is_main_process() and checkpointer is not None:
            log.info("Saving final checkpoint...")
            checkpoint_data = trainer.get_checkpoint_data()
            checkpointer.save_checkpoint(checkpoint_data)

            checkpointer.save_model_only(
                model.state_dict(),
                {
                    "iteration": trainer.state.iteration,
                    "val_loss": trainer.state.best_val_loss,
                    "config": trainer_config.llm.model_dump(),
                },
            )

        trainer.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 model")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    trainer_config = loader.load_training_config(args.config, gpt.GPT2Config)
    execution_strategy = trainer_config.training.execution.strategy

    launcher.setup_multiprocessing()

    if execution_strategy == execution_configs.ExecutionStrategy.SINGLE:
        main(trainer_config)

    else:  # DDP
        if "RANK" not in os.environ:
            raise RuntimeError(
                f"{execution_strategy.value} execution requires running with torchrun."
            )

        launcher.init_ddp_process_group()

        try:
            main(trainer_config)
        finally:
            distributed.cleanup_ddp()
