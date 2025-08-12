# Standard Library
import pathlib
import shutil
import time
import typing

# Third Party
import torch
import wandb
from torch import nn
from torch.distributed import fsdp
from torch.nn import parallel

# Project
from pretraining.common.base import inputs
from pretraining.configs import core
from pretraining.configs.training import execution_configs
from pretraining.configs.training import precision_configs
from pretraining.trainer import checkpoint_data
from pretraining.utils import logger as logger_utils
from pretraining.utils.training import dist_utils
from pretraining.utils.training import evaluation
from pretraining.utils.training import loss as loss_utils
from pretraining.utils.training import metrics
from pretraining.utils.training import state
from pretraining.utils.training.checkpointers import builder as checkpointer_builder


class LLMTrainer:
    """Trainer for Large Language Model pretraining.

    Core Operations:
    1. Initialize training components (model, optimizer, scheduler, data)
    2. Setup device placement and logging
    3. Load checkpoints if resuming training
    4. Run training loop:
       - Sample batches from infinite data stream
       - Forward pass through distributed model
       - Extract and aggregate losses (main + auxiliary)
       - Backward pass with gradient accumulation
       - Clip gradients and step optimizer
       - Track metrics (loss, perplexity, MFU, tokens/sec)
    5. Periodically evaluate on validation data
    6. Save checkpoints at intervals (rank 0 only)
    7. Handle mixed precision training
    8. Manage training state (iterations, tokens seen)
    9. Coordinate across GPUs for distributed training
    """

    def __init__(
        self,
        model: nn.Module,
        dist_model: typing.Union[nn.Module, parallel.DistributedDataParallel],
        config: core.TrainerConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_handler: loss_utils.LossHandler,
        evaluator: evaluation.Evaluator,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        val_dataloader: typing.Optional[torch.utils.data.DataLoader] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Initialize trainer with required components.

        Args:
            model: Base LLM model (for accessing methods/attributes)
            dist_model: Distributed model (DDP wrapped or single GPU model)
            config: Full trainer configuration (model + training)
            optimizer: Configured optimizer
            scheduler: Configured learning rate scheduler
            loss_handler: Loss handler for computing losses
            evaluator: Evaluator for computing validation metrics
            train_dataloader: Training data loader
            device: Device to run on
            val_dataloader: Optional validation data loader
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.model = model
        self.dist_model = dist_model
        self.config = config
        self.training_config = config.training  # For easy access to training config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_handler = loss_handler
        self.evaluator = evaluator
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0

        # Checkpoint tracking
        self.checkpoints: typing.List[pathlib.Path] = []
        self.last_checkpoint_step: typing.Optional[int] = None

        # Will be initialized in setup()
        self.logger = None
        self.state = None
        self.num_params = None

    def setup(self) -> None:
        """Setup trainer components."""
        self.logger = logger_utils.get_logger(__name__)

        self.state = state.TrainingState(
            max_iterations=self.training_config.max_iters,
            token_budget=self.training_config.token_budget,
            eval_interval=self.training_config.evaluation.eval_interval,
            checkpoint_interval=self.training_config.checkpoint.save_interval,
            log_interval=self.training_config.log_interval,
        )

        self.num_params = sum(p.numel() for p in self.model.parameters())

        # Initialize wandb if enabled
        if self.training_config.wandb_logging.enabled and (
            not self.training_config.wandb_logging.rank_zero_only or self.is_main_process
        ):
            # Create wandb directory
            wandb_dir = pathlib.Path(self.training_config.checkpoint.save_dir) / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)

            # Initialize wandb
            wandb.init(
                dir=str(wandb_dir),
                project=self.training_config.wandb_logging.project,
                entity=self.training_config.wandb_logging.entity,
                group=self.training_config.wandb_logging.group,
                name=self.training_config.wandb_logging.name,
                tags=self.training_config.wandb_logging.tags,
                config=self.config.model_dump(),  # Convert full config to dict
            )

            # Log model info
            wandb.config.update(
                {
                    "num_parameters": self.num_params,
                    "world_size": self.world_size,
                }
            )

        if self.is_main_process:
            self.logger.info(f"Trainer setup complete on {self.device}")
            self.logger.info(f"Model has {self.num_params:,} parameters")
            if self.world_size > 1:
                self.logger.info(f"Using distributed training with {self.world_size} processes")
            if self.training_config.wandb_logging.enabled:
                self.logger.info("Weights & Biases logging enabled")

    def train_step(self, batch: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, float]:
        """Run single training step.

        Args:
            batch: Dictionary with 'input_ids' and 'labels' from collator

        Returns:
            Dictionary of metrics for this step
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Create TrainingInputs from batch
        training_inputs = inputs.TrainingInputs(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch.get("attention_mask"),
            mtp_targets=batch.get("mtp_targets"),
        )

        # Mixed precision context
        use_amp = self.training_config.precision in [
            precision_configs.PrecisionType.AMP_BF16,
            precision_configs.PrecisionType.AMP_FP16,
        ]

        if use_amp:
            autocast_dtype = (
                torch.bfloat16
                if self.training_config.precision == precision_configs.PrecisionType.AMP_BF16
                else torch.float16
            )
            with torch.cuda.amp.autocast(enabled=True, dtype=autocast_dtype):
                # Forward pass through distributed model
                # The model will automatically use self.training to determine behavior
                model_output = self.dist_model.forward(
                    input_ids=training_inputs.input_ids,
                    attention_mask=training_inputs.attention_mask,
                )

                # Compute losses from logits
                losses = self.loss_handler.compute_losses(model_output, training_inputs)

                # Aggregate losses
                total_loss = self.loss_handler.aggregate_losses(losses)

                # Scale for gradient accumulation
                scaled_loss = total_loss / self.training_config.gradient_accumulation_steps
        else:
            # No autocast for fp32
            # Forward pass through distributed model
            # The model will automatically use self.training to determine behavior
            model_output = self.dist_model.forward(
                input_ids=training_inputs.input_ids,
                attention_mask=training_inputs.attention_mask,
            )

            # Compute losses from logits
            losses = self.loss_handler.compute_losses(model_output, training_inputs)

            # Aggregate losses
            total_loss = self.loss_handler.aggregate_losses(losses)

            # Scale for gradient accumulation
            scaled_loss = total_loss / self.training_config.gradient_accumulation_steps

        # Backward pass
        scaled_loss.backward()

        # Collect step metrics
        step_metrics = metrics.collect_train_step_metrics(
            loss=total_loss,
            ce_loss=losses.get("ce_loss"),
            z_loss=losses.get("z_loss"),
            moe_aux_loss=losses.get("moe_aux_loss"),
            mtp_losses=losses.get("mtp_losses"),
        )

        return step_metrics

    def evaluate(self) -> typing.Dict[str, float]:
        """Run evaluation on validation set.

        This method manages the model's train/eval state to ensure
        proper behavior during evaluation.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.val_dataloader is None:
            return {}

        # Save current training state and switch to eval mode
        was_training = self.dist_model.training
        self.dist_model.eval()

        try:
            # Run evaluation
            eval_metrics = self.evaluator.evaluate(
                self.dist_model,
                self.val_dataloader,
            )
            return eval_metrics
        finally:
            # Always restore original training state
            if was_training:
                self.dist_model.train()

    def optimizer_step(self) -> typing.Dict[str, float]:
        """Run optimizer step with gradient clipping.

        Returns:
            Dictionary with gradient norm
        """

        # Compute gradient norm before clipping
        grad_norm = metrics.compute_gradient_norm(self.dist_model)

        # Clip gradients
        if self.training_config.optimizer.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.dist_model.parameters(), self.training_config.optimizer.grad_clip
            )

        # Optimizer step
        self.optimizer.step()

        # Clear gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Step scheduler
        self.scheduler.step()

        # Collect optimizer metrics
        optim_metrics = metrics.collect_optimizer_metrics(self.optimizer, grad_norm)
        return optim_metrics

    def train(self) -> None:
        """Run training loop."""
        if self.logger is None:
            raise RuntimeError("Trainer not setup. Call setup() first.")

        if self.is_main_process:
            self.logger.info("Starting training")

        # Training loop
        self.dist_model.train()

        # Create iterator from dataloader
        train_iter = iter(self.train_dataloader)

        while not self.state.should_stop():
            # Start iteration timer
            iter_start_time = time.time()

            # Initialize accumulators for metrics across micro-batches
            accumulated_metrics = {}

            # Accumulate gradients over multiple batches
            for micro_batch_idx in range(self.training_config.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    # Restart iterator when exhausted
                    train_iter = iter(self.train_dataloader)
                    batch = next(train_iter)

                step_metrics = self.train_step(batch)

                # Accumulate metrics across micro-batches
                for key, value in step_metrics.items():
                    if key not in accumulated_metrics:
                        accumulated_metrics[key] = 0.0
                    accumulated_metrics[key] += value

            # Average accumulated metrics
            for key in accumulated_metrics:
                accumulated_metrics[key] /= self.training_config.gradient_accumulation_steps

            # Optimizer step
            optim_metrics = self.optimizer_step()

            # Update state
            batch_size = (
                batch["input_ids"].shape[0] * self.training_config.gradient_accumulation_steps
            )
            seq_length = batch["input_ids"].shape[1]
            self.state.update(batch_size, seq_length)

            # Compute throughput metrics
            iter_time = time.time() - iter_start_time
            tokens_processed = batch_size * seq_length
            # tokens_per_sec = tokens_processed / iter_time

            # Get GPU memory usage
            # peak_memory_mb = torch_utils.peak_gpu_memory(reset=False)

            # Collect all metrics for this step
            if self.state.should_log():
                # Start with accumulated training metrics
                all_metrics = accumulated_metrics.copy()

                # Add optimizer metrics
                all_metrics.update(optim_metrics)

                # Add throughput metrics
                throughput_metrics = metrics.collect_throughput_metrics(
                    tokens_processed=tokens_processed,
                    elapsed_time=iter_time,
                    global_step=self.state.iteration,
                )
                all_metrics.update(throughput_metrics)

                # Add system metrics
                system_metrics = metrics.collect_system_metrics()
                all_metrics.update(system_metrics)

                # Add progress info
                all_metrics.update(self.state.get_progress())

                # Log to console (main process only)
                if self.is_main_process:
                    # Format for logging
                    log_message = (
                        f"[Step {self.state.iteration}] "
                        f"loss: {accumulated_metrics.get('train/ce_loss', accumulated_metrics.get('train/total_loss', 0)):.4f}, "
                        f"ppl: {accumulated_metrics.get('train/perplexity', 0):.2f}, "
                        f"lr: {optim_metrics.get('optim/lr', 0):.2e}, "
                        f"tokens/s: {throughput_metrics.get('throughput/tokens_per_sec', 0):.0f}"
                    )
                    if "system/peak_gpu_memory_mb" in system_metrics:
                        log_message += f", mem: {system_metrics['system/peak_gpu_memory_mb']:.0f}MB"
                    self.logger.info(log_message)

                # Log to W&B if enabled
                if self.training_config.wandb_logging.enabled and (
                    not self.training_config.wandb_logging.rank_zero_only or self.is_main_process
                ):
                    if self.state.iteration % self.training_config.wandb_logging.log_interval == 0:
                        wandb.log(all_metrics, step=self.state.iteration)

            # Evaluation
            if self.state.should_eval():
                if self.is_main_process:
                    self.logger.info("Running evaluation...")

                eval_metrics = self.evaluate()

                # Check if best model
                val_loss = eval_metrics["val/loss"]
                is_best = val_loss < self.state.best_val_loss

                if self.is_main_process:
                    self.logger.info(
                        f"[Eval] loss: {val_loss:.4f}, "
                        f"ppl: {eval_metrics['val/perplexity']:.2f}"
                        + (" (best)" if is_best else "")
                    )

                # Update state
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss

                # Log eval metrics to W&B
                if self.training_config.wandb_logging.enabled and (
                    not self.training_config.wandb_logging.rank_zero_only or self.is_main_process
                ):
                    wandb.log(eval_metrics, step=self.state.iteration)

            # Checkpoint saving
            if self.state.should_checkpoint():
                if self.is_main_process:
                    self.logger.info(f"Saving checkpoint at step {self.state.iteration}...")

                checkpoint_path = self.save_checkpoint()

                if self.is_main_process:
                    self.logger.info(f"Checkpoint saved to {checkpoint_path}")

            # Synchronize across processes
            dist_utils.barrier()

        if self.is_main_process:
            self.logger.info(f"Training complete after {self.state.iteration} steps")

    def get_checkpoint_data(self) -> checkpoint_data.CheckpointData:
        """Get all data needed for checkpointing.

        Returns:
            CheckpointData object with all trainer state
        """
        # For FSDP, we need to use dist_model to get the proper sharded state
        # For DDP/Single, we can use either model or dist_model (they're equivalent)
        # We'll use dist_model for consistency since it works for all cases

        # Check if dist_model is FSDP-wrapped
        is_fsdp = isinstance(self.dist_model, fsdp.FullyShardedDataParallel)

        # Use dist_model for FSDP to get sharded state, model for others
        # Note: For FSDP checkpointers, these state dicts are often ignored
        # and re-extracted within proper FSDP contexts, but we provide them
        # for compatibility with non-FSDP checkpointers
        model_state = self.dist_model.state_dict() if is_fsdp else self.model.state_dict()

        return checkpoint_data.CheckpointData(
            model_state=model_state,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            training_state=self.state.get_checkpoint_dict(),
        )

    def load_checkpoint_data(self, ckpt_data: checkpoint_data.CheckpointData) -> None:
        """Load state from checkpoint data.

        Args:
            ckpt_data: CheckpointData object to load from
        """
        # Check if dist_model is FSDP-wrapped
        is_fsdp = isinstance(self.dist_model, fsdp.FullyShardedDataParallel)

        # Load model state
        # For FSDP, the checkpointer handles loading within FSDP context
        # This is mainly for non-FSDP cases
        if not is_fsdp:
            self.model.load_state_dict(ckpt_data.model_state)
            self.optimizer.load_state_dict(ckpt_data.optimizer_state)
        # For FSDP, the checkpointer already loaded states in its load_checkpoint method

        # Scheduler state can be loaded for all cases
        self.scheduler.load_state_dict(ckpt_data.scheduler_state)

        # Load training state
        self.state.load_checkpoint_dict(ckpt_data.training_state)

        if self.is_main_process:
            self.logger.info(f"Loaded checkpoint at step {self.state.iteration}")

    def cleanup(self) -> None:
        """Clean up resources after training."""
        # Finish wandb run if it was initialized
        if self.training_config.wandb_logging.enabled and (
            not self.training_config.wandb_logging.rank_zero_only or self.is_main_process
        ):
            wandb.finish()

        if self.is_main_process:
            self.logger.info("Trainer cleanup complete")

    def save_checkpoint(self) -> pathlib.Path:
        """Save a training checkpoint.

        Creates checkpointer on-demand and saves checkpoint.
        All ranks participate but with different roles based on strategy.

        Returns:
            Path to saved checkpoint directory
        """
        # Zero gradients to avoid saving them
        self.optimizer.zero_grad(set_to_none=True)

        # Build checkpointer based on execution strategy
        checkpointer = checkpointer_builder.build_checkpointer(
            config=self.training_config.checkpoint,
            execution=self.training_config.execution,
            dist_model=self.dist_model
            if self.training_config.execution.strategy == execution_configs.ExecutionStrategy.FSDP
            else None,
            optimizer=self.optimizer
            if self.training_config.execution.strategy == execution_configs.ExecutionStrategy.FSDP
            else None,
        )

        # Get checkpoint data
        ckpt_data = self.get_checkpoint_data()

        # Save checkpoint (all ranks participate for FSDP, rank 0 only for others)
        if (
            self.training_config.execution.strategy == execution_configs.ExecutionStrategy.FSDP
            or self.is_main_process
        ):
            checkpointer.save_checkpoint(ckpt_data, self.config)

        # Sync across ranks
        dist_utils.barrier()

        # Track checkpoint (main process only)
        checkpoint_dir = (
            pathlib.Path(self.training_config.checkpoint.save_dir) / f"step{self.state.iteration}"
        )
        if self.is_main_process:
            self.checkpoints.append(checkpoint_dir)
            self.last_checkpoint_step = self.state.iteration

            # Rotate old checkpoints
            self._rotate_checkpoints()

        return checkpoint_dir

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to keep only the last N."""
        keep_last_n = self.training_config.checkpoint.keep_last_n
        if keep_last_n <= 0:
            return  # Keep all checkpoints

        while len(self.checkpoints) > keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)

            # Sync before removal
            dist_utils.barrier()

            if old_checkpoint.exists() and self.is_main_process:
                self.logger.info(f"Removing old checkpoint: {old_checkpoint}")
                shutil.rmtree(old_checkpoint)

            # Sync after removal
            dist_utils.barrier()

    def load_checkpoint(self, checkpoint_path: typing.Optional[pathlib.Path] = None) -> bool:
        """Load a checkpoint for resuming training.

        Args:
            checkpoint_path: Specific checkpoint to load, or None to find latest

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        # Build checkpointer
        checkpointer = checkpointer_builder.build_checkpointer(
            config=self.training_config.checkpoint,
            execution=self.training_config.execution,
            dist_model=self.dist_model
            if self.training_config.execution.strategy == execution_configs.ExecutionStrategy.FSDP
            else None,
            optimizer=self.optimizer
            if self.training_config.execution.strategy == execution_configs.ExecutionStrategy.FSDP
            else None,
        )

        # Override resume_from if specific path provided
        if checkpoint_path:
            checkpointer.config.resume_from = str(checkpoint_path)

        # Load checkpoint (all ranks for FSDP, rank 0 for others)
        ckpt_data = None
        if self.training_config.execution.strategy == execution_configs.ExecutionStrategy.FSDP:
            ckpt_data = checkpointer.load_checkpoint(str(self.device))
        elif self.is_main_process:
            ckpt_data = checkpointer.load_checkpoint(str(self.device))

        # Sync across ranks
        dist_utils.barrier()

        if ckpt_data is not None:
            self.load_checkpoint_data(ckpt_data)
            return True

        return False
