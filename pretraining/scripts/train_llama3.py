# Standard Library
import argparse
import os

# Third Party
import torch

# Project
from pretraining.common.models.architectures import llama3
from pretraining.configs import core
from pretraining.configs import loader
from pretraining.configs.model.architectures import llama
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
from pretraining.utils.training import optimizer
from pretraining.utils.training import setup


def main(trainer_config: core.TrainerConfig[llama.Llama3Config]) -> None:
    """Main training function that runs after distributed setup."""
    device = trainer_config.training.execution.setup_device()

    logger_utils.setup_logging(rank=dist_utils.get_global_rank())
    log = logger_utils.get_logger(__name__)

    log.info(f"Using device: {device}")
    log.info(f"Execution strategy: {trainer_config.training.execution.strategy.value}")

    torch_utils.seed_all(trainer_config.training.seed)

    log.info("Building Llama 3 model...")
    model = llama3.Llama3.from_config(trainer_config.llm)

    # (device placement, wrapping, initialization)
    dist_model = setup.prepare_model_for_training(model, trainer_config, device)

    # Create AdamW optimizer with dimension-based parameter grouping (Llama style)
    optim = optimizer.OptimizerFactory.create_adamw(
        model=dist_model,
        learning_rate=trainer_config.training.optimizer.learning_rate,
        weight_decay=trainer_config.training.optimizer.weight_decay,
        betas=(trainer_config.training.optimizer.beta1, trainer_config.training.optimizer.beta2),
        eps=trainer_config.training.optimizer.eps,
        parameter_grouping=trainer_config.training.optimizer.parameter_grouping,
        no_decay_patterns=trainer_config.training.optimizer.no_decay_patterns,
        device_type=device.type,
    )

    # Create cosine schedule with warmup (Llama uses cosine decay)
    scheduler = lr_scheduler.LRSchedulerFactory.create_cosine_with_warmup(
        optimizer=optim,
        num_warmup_steps=trainer_config.training.lr_schedule.warmup_iters,
        num_training_steps=trainer_config.training.max_iters,
        num_cycles=trainer_config.training.lr_schedule.num_cycles,
    )

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

    start_index, epoch = setup.load_checkpoint_if_resuming(trainer, trainer_config)

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
        setup.save_final_checkpoint(trainer)
        trainer.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Llama 3 model")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="pretraining/configs/examples/debug/llama3_debug_cpu.yaml",
        help="Path to YAML config file (defaults to debug config)",
    )
    args = parser.parse_args()

    trainer_config = loader.load_training_config(args.config, llama.Llama3Config)
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
