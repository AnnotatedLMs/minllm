# Standard Library
import pathlib
import time
import typing

# Third Party
import torch
import torch.nn as nn
import wandb
from torch.nn import parallel

# Project
from pretraining.common.base import inputs
from pretraining.configs.training import trainer_configs
from pretraining.trainer import checkpoint_data as checkpoint_data_module
from pretraining.utils import logger as logger_utils
from pretraining.utils import torch_utils
from pretraining.utils.training import evaluation
from pretraining.utils.training import loss as loss_utils
from pretraining.utils.training import metrics
from pretraining.utils.training import state


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
        config: trainer_configs.TrainingLoopConfig,
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
            config: Training configuration
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

        # Will be initialized in setup()
        self.logger = None
        self.state = None
        self.scaler = None
        self.num_params = None

    def setup(self) -> None:
        """Setup trainer components."""
        self.logger = logger_utils.get_logger(__name__)

        self.state = state.TrainingState(
            max_iterations=self.config.max_iters,
            token_budget=self.config.token_budget,
            eval_interval=self.config.evaluation.eval_interval,
            checkpoint_interval=self.config.checkpoint.save_interval,
            log_interval=self.config.log_interval,
        )

        self.scaler = torch.amp.GradScaler() if self.config.mixed_precision else None

        self.num_params = sum(p.numel() for p in self.model.parameters())

        # Initialize wandb if enabled
        if self.config.wandb_logging.enabled and (
            not self.config.wandb_logging.rank_zero_only or self.is_main_process
        ):
            # Create wandb directory
            wandb_dir = pathlib.Path(self.config.checkpoint.checkpoint_dir) / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)

            # Initialize wandb
            wandb.init(
                dir=str(wandb_dir),
                project=self.config.wandb_logging.project,
                entity=self.config.wandb_logging.entity,
                group=self.config.wandb_logging.group,
                name=self.config.wandb_logging.name,
                tags=self.config.wandb_logging.tags,
                config=self.config.model_dump(),  # Convert config to dict
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
            if self.config.wandb_logging.enabled:
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
        autocast_context = torch.cuda.amp.autocast if self.config.mixed_precision else torch.no_grad

        with autocast_context(enabled=self.config.mixed_precision):
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
            scaled_loss = total_loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
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
        # Unscale gradients if using AMP
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Compute gradient norm before clipping
        grad_norm = metrics.compute_gradient_norm(self.dist_model)

        # Clip gradients
        if self.config.optimizer.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.dist_model.parameters(), self.config.optimizer.grad_clip
            )

        # Optimizer step
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
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
            for micro_batch_idx in range(self.config.gradient_accumulation_steps):
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
                accumulated_metrics[key] /= self.config.gradient_accumulation_steps

            # Optimizer step
            optim_metrics = self.optimizer_step()

            # Update state
            batch_size = batch["input_ids"].shape[0] * self.config.gradient_accumulation_steps
            seq_length = batch["input_ids"].shape[1]
            self.state.update(batch_size, seq_length)

            # Compute throughput metrics
            iter_time = time.time() - iter_start_time
            tokens_processed = batch_size * seq_length
            tokens_per_sec = tokens_processed / iter_time

            # Get GPU memory usage
            peak_memory_mb = torch_utils.peak_gpu_memory(reset=False)

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
                if self.config.wandb_logging.enabled and (
                    not self.config.wandb_logging.rank_zero_only or self.is_main_process
                ):
                    if self.state.iteration % self.config.wandb_logging.log_interval == 0:
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
                if self.config.wandb_logging.enabled and (
                    not self.config.wandb_logging.rank_zero_only or self.is_main_process
                ):
                    wandb.log(eval_metrics, step=self.state.iteration)

            # Synchronize across processes
            torch_utils.barrier()

        if self.is_main_process:
            self.logger.info(f"Training complete after {self.state.iteration} steps")

    def get_checkpoint_data(self) -> checkpoint_data_module.CheckpointData:
        """Get all data needed for checkpointing.

        Returns:
            CheckpointData object with all trainer state
        """
        return checkpoint_data_module.CheckpointData(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            scaler_state_dict=self.scaler.state_dict() if self.scaler else None,
            training_state_dict=self.state.get_checkpoint_dict(),
            config=self.config,
            model_args={},  # Could be populated if needed for model recreation
        )

    def load_checkpoint_data(self, checkpoint_data: checkpoint_data_module.CheckpointData) -> None:
        """Load state from checkpoint data.

        Args:
            checkpoint_data: CheckpointData object to load from
        """
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint_data.model_state_dict)
        self.optimizer.load_state_dict(checkpoint_data.optimizer_state_dict)
        self.scheduler.load_state_dict(checkpoint_data.scheduler_state_dict)

        if self.scaler and checkpoint_data.scaler_state_dict:
            self.scaler.load_state_dict(checkpoint_data.scaler_state_dict)

        # Load training state
        self.state.load_checkpoint_dict(checkpoint_data.training_state_dict)

        if self.is_main_process:
            self.logger.info(f"Loaded checkpoint at step {self.state.iteration}")

    def cleanup(self) -> None:
        """Clean up resources after training."""
        # Finish wandb run if it was initialized
        if self.config.wandb_logging.enabled and (
            not self.config.wandb_logging.rank_zero_only or self.is_main_process
        ):
            wandb.finish()

        if self.is_main_process:
            self.logger.info("Trainer cleanup complete")
