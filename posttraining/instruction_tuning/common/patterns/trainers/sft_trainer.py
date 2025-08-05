# Standard Library
import math
import os
import typing

# Third Party
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

# Project
from posttraining.instruction_tuning.common.patterns.trainers import training_steps
from posttraining.instruction_tuning.utils import checkpoint_utils
from posttraining.instruction_tuning.utils import metrics

logger = get_logger(__name__)


class SFTTrainer:
    """Trainer for Supervised Fine-Tuning (SFT) of language models.

    Core Concept:
    SFT adapts a pretrained language model to follow instructions by training on
    (instruction, response) pairs. Unlike pretraining which predicts all tokens,
    SFT only computes loss on response tokens, teaching the model to generate
    appropriate responses to given instructions.

    Key Differences from Pretraining:
    1. Selective Loss: Only response tokens contribute to loss (via label masking)
    2. Smaller Scale: Typically uses much less data and compute than pretraining
    3. Chat Format: Uses chat templates to structure conversations
    4. Preservation: Aims to maintain general capabilities while adding instruction-following

    Training Process:
    1. Load pretrained model (e.g., base Llama, GPT)
    2. Format data with chat templates (system, user, assistant turns)
    3. Tokenize and mask non-response tokens with -100
    4. Train with cross-entropy loss only on response tokens
    5. Use techniques like LoRA for parameter-efficient training

    Common Techniques:
    - LoRA/QLoRA: Train only low-rank adaptations to save memory
    - Gradient Accumulation: Simulate larger batches on limited hardware
    - Mixed Precision: Use fp16/bf16 for efficiency
    - Packing: Concatenate short examples to maximize GPU utilization

    The result is a model that maintains its language modeling capabilities
    while becoming much better at following instructions and engaging in
    helpful dialogue.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataloader: DataLoader,
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: typing.Any,
        args: typing.Any,  # SFTArguments
    ):
        """
        Initialize the SFT trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataloader: Training data loader
            accelerator: Accelerator instance
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            args: Training arguments
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.args = args

        # Get vocabulary size for loss computation
        self.vocab_size = model.config.vocab_size

        # Initialize metrics tracker
        self.metrics_tracker = metrics.SFTMetricsTracker(
            accelerator=accelerator,
            logging_steps=args.logging_steps,
            per_device_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            reduce_loss=args.reduce_loss,
        )

    def train(self):
        """Run the training loop."""
        # Calculate training steps
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps
        )

        # Handle max_train_steps
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch

        # Recalculate epochs based on max steps
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )

        # Log training info
        total_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}"
        )
        logger.info(f"  Total train batch size = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        # Initialize progress tracking
        progress_bar = tqdm(
            range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process
        )
        completed_steps = 0
        starting_epoch = 0
        resume_batch_idx = 0

        # Handle checkpoint resumption
        if self.args.resume_from_checkpoint:
            self._resume_from_checkpoint()
            starting_epoch, resume_batch_idx, _, completed_steps = self._get_resume_info()
            progress_bar.update(completed_steps)

        # Training loop
        for epoch in range(starting_epoch, self.args.num_train_epochs):
            self._train_epoch(
                epoch,
                completed_steps,
                progress_bar,
                resume_batch_idx if epoch == starting_epoch else 0,
            )

            # Save epoch checkpoint if requested
            if self.args.checkpointing_steps == "epoch":
                self._save_checkpoint(f"epoch_{epoch}")

            # Check if we've reached max steps
            if completed_steps >= self.args.max_train_steps:
                break

        # Save final model
        if self.args.output_dir is not None:
            checkpoint_utils.save_model_checkpoint(
                self.accelerator,
                self.model,
                self.tokenizer,
                self.args.output_dir,
                self.args.use_lora,
            )

        # Clean checkpoints if requested
        if self.accelerator.is_local_main_process and hasattr(
            self.args, "clean_checkpoints_at_end"
        ):
            if self.args.clean_checkpoints_at_end:
                checkpoint_utils.clean_checkpoints(self.args.output_dir, keep_last_n_checkpoints=0)

    def _train_epoch(
        self, epoch: int, completed_steps: int, progress_bar: tqdm, resume_batch_idx: int = 0
    ) -> int:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
            completed_steps: Number of completed optimization steps
            progress_bar: Progress bar to update
            resume_batch_idx: Batch index to resume from

        Returns:
            Updated completed_steps
        """
        self.model.train()

        # Prepare dataloader
        active_dataloader = training_steps.prepare_dataloader_for_epoch(
            self.train_dataloader,
            epoch,
            resume_batch_idx,
            self.accelerator,
        )

        # Training loop
        for step, batch in enumerate(active_dataloader):
            # Update batch metrics
            self.metrics_tracker.update_batch_metrics(batch)

            # Forward pass with gradient accumulation
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = training_steps.perform_forward_pass(self.model, batch, use_cache=False)

                # Compute loss
                loss = training_steps.compute_training_loss(
                    outputs,
                    batch,
                    self.vocab_size,
                    self.args.reduce_loss,
                )

                # Track loss
                self.metrics_tracker.update_loss(loss)

                # Backward pass and gradient clipping
                training_steps.perform_backward_pass(
                    self.accelerator,
                    loss,
                    self.model,
                    self.args.clip_grad_norm,
                )

                # Optimizer step
                training_steps.perform_optimizer_step(self.optimizer, self.lr_scheduler)

            # Update progress if we did an optimization step
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                # Log metrics
                if training_steps.should_log_metrics(completed_steps, self.args.logging_steps):
                    training_steps.log_training_metrics(
                        self.metrics_tracker,
                        completed_steps,
                        self.lr_scheduler,
                        self.accelerator,
                        self.args.report_to,
                    )

                # Save checkpoint
                if training_steps.should_save_checkpoint(
                    completed_steps, self.args.checkpointing_steps
                ):
                    self._save_checkpoint(f"step_{completed_steps}")

                # Check if we've reached max steps
                if completed_steps >= self.args.max_train_steps:
                    break

        return completed_steps

    def _save_checkpoint(self, checkpoint_name: str):
        """Save a training checkpoint."""
        output_dir = os.path.join(self.args.output_dir, checkpoint_name)
        self.accelerator.save_state(output_dir)

        # Mark checkpoint as complete
        if self.accelerator.is_main_process:
            with open(os.path.join(output_dir, "COMPLETED"), "w") as f:
                f.write("COMPLETED")

            # Clean old checkpoints
            if hasattr(self.args, "keep_last_n_checkpoints"):
                checkpoint_utils.clean_checkpoints(
                    self.args.output_dir, self.args.keep_last_n_checkpoints
                )

        self.accelerator.wait_for_everyone()

    def _resume_from_checkpoint(self):
        """Resume training from checkpoint."""
        checkpoint_path = checkpoint_utils.get_last_checkpoint_path(self.args.output_dir)
        if checkpoint_path:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            self.accelerator.load_state(checkpoint_path)

    def _get_resume_info(self) -> typing.Tuple[int, int, int, int]:
        """Get resume information from checkpoint."""
        checkpoint_path = checkpoint_utils.get_last_checkpoint_path(self.args.output_dir)
        if not checkpoint_path:
            return 0, 0, 0, 0

        return checkpoint_utils.parse_checkpoint_step(checkpoint_path)
