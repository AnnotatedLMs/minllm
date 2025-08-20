# Standard Library
import argparse
import os

# Third Party
import torch
from packaging import version

# Project
from pretraining.common.models.architectures import deepseek3
from pretraining.configs import core
from pretraining.configs import loader
from pretraining.configs.model.architectures import deepseek
from pretraining.configs.training import execution_configs
from pretraining.data import builder
from pretraining.data import collator
from pretraining.trainer import llm_trainer
from pretraining.utils import logger as logger_utils
from pretraining.utils import torch_utils
from pretraining.utils.training import dist_utils
from pretraining.utils.training import evaluation
from pretraining.utils.training import launcher
from pretraining.utils.training import loss
from pretraining.utils.training import lr_scheduler
from pretraining.utils.training import model_wrapper
from pretraining.utils.training import optimizer
from pretraining.utils.training.checkpointers import checkpointer_factory


def main(trainer_config: core.TrainerConfig[deepseek.DeepSeek3Config]) -> None:
    """Main training function that runs after distributed setup."""
    device = trainer_config.training.execution.setup_device()

    logger_utils.setup_logging(rank=dist_utils.get_global_rank())
    log = logger_utils.get_logger(__name__)

    log.info(f"Using device: {device}")
    log.info(f"Execution strategy: {trainer_config.training.execution.strategy.value}")

    torch_utils.seed_all(trainer_config.training.seed)

    log.info("Building DeepSeek 3 model...")
    model = deepseek3.DeepSeek3.from_config(trainer_config.llm)

    # Move model to device (FSDP handles this internally)
    if trainer_config.training.execution.strategy != execution_configs.ExecutionStrategy.FSDP:
        model = model.to(device)

    # Wrap model based on execution strategy
    log.info(f"Wrapping model with {trainer_config.training.execution.strategy.value}...")
    dist_model = model_wrapper.wrap_model(
        model,
        trainer_config.training.execution,
        trainer_config.training.precision_dtype,
    )

    # Initialize parameters if needed (following OLMo's pattern)
    # When param_init_fn is None, FSDP will call reset_parameters() automatically
    # For DDP or when using PyTorch >= 2.1.0 with FSDP, we need to call it manually
    if trainer_config.training.execution.strategy == execution_configs.ExecutionStrategy.DDP:
        model.reset_parameters()
    elif trainer_config.training.execution.strategy == execution_configs.ExecutionStrategy.FSDP:
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            model.reset_parameters()

    optim = optimizer.OptimizerFactory.create_from_config(
        dist_model, trainer_config.training.optimizer, device_type=device.type
    )

    scheduler = lr_scheduler.LRSchedulerFactory.create_from_config(
        optim,
        trainer_config.training.lr_schedule,
        num_training_steps=trainer_config.training.max_iters,
    )

    # We'll load checkpoint after creating trainer
    start_index = 0
    epoch = 0

    log.info("Setting up data pipeline...")

    train_dataset = builder.build_memmap_dataset(
        data_config=trainer_config.training.data,
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

    data_collator = collator.MTPDataCollator(mtp_depth=trainer_config.llm.mtp.n_predict)

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
        data_config=trainer_config.training.data,
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
        collate_fn=collator.DataCollator(),
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
        config=trainer_config,
        optimizer=optim,
        scheduler=scheduler,
        loss_handler=loss_handler,
        evaluator=evaluator,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        rank=dist_utils.get_global_rank(),
        world_size=dist_utils.get_world_size(),
    )

    trainer.setup()

    # Try to load checkpoint if resuming
    should_resume, resume_path = checkpointer_factory.should_resume_training(
        trainer_config.training.checkpoint
    )
    if should_resume:
        log.info(f"Attempting to resume from {resume_path}")
        if trainer.load_checkpoint():
            log.info(f"Successfully resumed from iteration {trainer.state.iteration}")
            # Update start_index and epoch from loaded state
            if hasattr(trainer.state, "training_state"):
                start_index = trainer.state.training_state.get("tokens_seen", 0)
                epoch = trainer.state.training_state.get("epoch", 0)
        else:
            log.warning(f"Failed to load checkpoint from {resume_path}, starting from scratch")

    # Synchronize all ranks before starting training
    dist_utils.barrier()

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
        # Save final checkpoint only if we haven't just saved at this step
        if trainer.last_checkpoint_step != trainer.state.iteration:
            if dist_utils.is_main_process():
                log.info("Saving final checkpoint...")
            checkpoint_path = trainer.save_checkpoint()
            if dist_utils.is_main_process():
                log.info(f"Final checkpoint saved to {checkpoint_path}")
        else:
            if dist_utils.is_main_process():
                log.info(f"Final checkpoint already saved at step {trainer.state.iteration}")

        # Clean up trainer resources (including wandb)
        trainer.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepSeek 3 model")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="pretraining/configs/examples/debug/deepseek3_debug_cpu.yaml",
        help="Path to YAML config file (defaults to debug config)",
    )
    args = parser.parse_args()

    trainer_config = loader.load_training_config(args.config, deepseek.DeepSeek3Config)
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
            dist_utils.cleanup_ddp()
