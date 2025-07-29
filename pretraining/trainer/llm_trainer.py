# Standard Library
import time
import typing

# Third Party
import torch
import torch.nn as nn
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
        dist_model: typing.Union[
            nn.Module, parallel.DistributedDataParallel
        ],  # TODO extend to FSDP later
        config: trainer_configs.TrainingLoopConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
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
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0

        # Will be initialized in setup()
        self.logger = None
        self.state = None
        self.metrics_tracker = None
        self.scaler = None
        self.num_params = None

    def setup(self) -> None:
        """Setup trainer components."""
        # Setup logging
        self.logger = logger_utils.get_logger(__name__)

        # Setup training state
        self.state = state.TrainingState(
            max_iterations=self.config.training.max_steps,
            max_tokens=self.config.training.max_tokens,
            eval_interval=self.config.evaluator.eval_interval,
            checkpoint_interval=self.config.checkpointer.save_interval,
            log_interval=self.config.training.log_interval,
        )

        # Setup metrics tracking
        self.metrics_tracker = metrics.MetricsTracker()

        # Setup mixed precision
        self.scaler = torch.amp.GradScaler() if self.config.training.mixed_precision else None

        # Calculate model info
        self.num_params = sum(p.numel() for p in self.model.parameters())

        if self.is_main_process:
            self.logger.info(f"Trainer setup complete on {self.device}")
            self.logger.info(f"Model has {self.num_params:,} parameters")
            if self.world_size > 1:
                self.logger.info(f"Using distributed training with {self.world_size} processes")

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
        autocast_context = (
            torch.cuda.amp.autocast if self.config.training.mixed_precision else torch.no_grad
        )

        with autocast_context(enabled=self.config.training.mixed_precision):
            # Forward pass through distributed model
            model_output = self.dist_model.training_forward(training_inputs)

            # Extract losses
            losses = loss_utils.LossHandler.extract_losses(model_output)

            # Aggregate losses
            total_loss = loss_utils.LossHandler.aggregate_losses(losses)

            # Scale for gradient accumulation
            scaled_loss = total_loss / self.config.training.gradient_accumulation_steps

        # Backward pass
        if self.config.training.mixed_precision and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Prepare metrics
        step_metrics = {f"train/{name}": value.item() for name, value in losses.items()}
        step_metrics["train/total_loss"] = total_loss.item()

        return step_metrics

    def optimizer_step(self) -> typing.Dict[str, float]:
        """Run optimizer step with gradient clipping.

        Returns:
            Dictionary with gradient norm
        """
        # Unscale gradients if using AMP
        if self.config.training.mixed_precision and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Compute gradient norm before clipping
        grad_norm = metrics.compute_gradient_norm(self.dist_model)

        # Clip gradients
        if self.config.training.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.dist_model.parameters(), self.config.training.max_grad_norm
            )

        # Optimizer step
        if self.config.training.mixed_precision and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Clear gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Step scheduler
        self.scheduler.step()

        return {"train/grad_norm": grad_norm}

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
            self.metrics_tracker.reset()

            # Accumulate gradients over multiple batches
            for _ in range(self.config.training.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    # Restart iterator when exhausted
                    train_iter = iter(self.train_dataloader)
                    batch = next(train_iter)

                step_metrics = self.train_step(batch)

                # Update metrics tracker
                for key, value in step_metrics.items():
                    self.metrics_tracker.update(**{key.split("/")[1]: value})

            # Optimizer step
            optim_metrics = self.optimizer_step()

            # Update state
            batch_size = (
                batch["input_ids"].shape[0] * self.config.training.gradient_accumulation_steps
            )
            seq_length = batch["input_ids"].shape[1]
            self.state.update(batch_size, seq_length)

            # Compute throughput metrics
            iter_time = time.time() - iter_start_time
            tokens_processed = batch_size * seq_length
            tokens_per_sec = tokens_processed / iter_time

            # Get GPU memory usage
            peak_memory_mb = torch_utils.peak_gpu_memory(reset=False)

            # Log metrics (main process only)
            if self.state.should_log() and self.is_main_process:
                log_metrics = self.metrics_tracker.get_all_averages()
                log_metrics.update(optim_metrics)
                log_metrics.update(
                    {
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    }
                )
                if peak_memory_mb is not None:
                    log_metrics["train/peak_gpu_memory_mb"] = peak_memory_mb
                log_metrics.update(self.state.get_progress())

                # Format for logging
                log_message = (
                    f"[Step {self.state.iteration}] "
                    f"loss: {log_metrics.get('loss', 0):.4f}, "
                    f"ppl: {metrics.compute_perplexity(log_metrics.get('loss', 0)):.2f}, "
                    f"lr: {log_metrics['train/lr']:.2e}, "
                    f"tokens/s: {tokens_per_sec:.0f}"
                )
                if peak_memory_mb is not None:
                    log_message += f", mem: {peak_memory_mb:.0f}MB"
                self.logger.info(log_message)

            # Evaluation
            if self.state.should_eval():
                if self.is_main_process:
                    self.logger.info("Running evaluation...")

                eval_metrics = evaluation.estimate_loss(
                    self.dist_model, self.val_dataloader, num_batches=50
                )

                # Check if best model
                val_loss = eval_metrics["val/loss"]
                is_best = self.metrics_tracker.update_best("val_loss", val_loss)

                if self.is_main_process:
                    self.logger.info(
                        f"[Eval] loss: {val_loss:.4f}, "
                        f"ppl: {eval_metrics['val/perplexity']:.2f}"
                        + (" (best)" if is_best else "")
                    )

                # Update state
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss

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
