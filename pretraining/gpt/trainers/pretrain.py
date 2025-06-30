# Standard Library
import contextlib
import logging
import pathlib
import time

# Third Party
import jaxtyping
import torch
import wandb
from torch.backends import cuda
from torch.backends import cudnn
from torch.nn import parallel

# Project
from pretraining.gpt.data import dataloader
from pretraining.gpt.models import gpt
from pretraining.gpt.utils import checkpointer
from pretraining.gpt.utils import config
from pretraining.gpt.utils import distribution
from pretraining.gpt.utils import lr_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the complete pretraining pipeline for GPT models.
    Designed for clarity and educational value.
    """

    def __init__(self, config: config.TrainerConfig):
        self.config = config

        # These will be initialized in setup methods
        self.model = None
        self.optimizer = None
        self.dataloader = None
        self.lr_scheduler = None
        self.checkpointer = None
        self.scaler = None

        # Training state
        self.iter_num = 0
        self.best_val_loss = 1e9

        # DDP state
        self.ddp_rank = 0
        self.ddp_local_rank = 0
        self.ddp_world_size = 1
        self.device = config.device

    def setup_distributed_training(self):
        """Initialize distributed training if running with torchrun."""
        self.ddp_rank, self.ddp_local_rank, self.ddp_world_size = distribution.setup_ddp(
            backend=self.config.backend
        )

        # Adjust gradient accumulation for DDP
        if self.ddp_world_size > 1:
            assert self.config.gradient_accumulation_steps % self.ddp_world_size == 0
            self.config.gradient_accumulation_steps //= self.ddp_world_size

    def setup_device(self):
        """Configure device and dtypes for training."""
        if self.ddp_world_size > 1:
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)

        # Configure matrix multiply precision
        cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True

        # Setup autocast context
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            self.config.dtype
        ]

        self.ctx = (
            contextlib.nullcontext()
            if self.config.device_type == "cpu"
            else torch.amp.autocast(device_type=self.config.device_type, dtype=ptdtype)
        )

    def setup_directories(self):
        """Create output directories on main process."""
        if distribution.is_main_process():
            pathlib.Path(self.config.out_dir).mkdir(parents=True, exist_ok=True)

    def set_seed(self):
        """Set random seed with offset for DDP."""
        torch.manual_seed(self.config.seed + self.ddp_rank)

    def init_model(self):
        """Initialize model based on init_from config."""
        logger.info(f"Initializing model from '{self.config.init_from}'")

        # Try to load vocab size from dataset metadata
        meta_path = pathlib.Path(self.config.data_dir) / self.config.dataset / "meta.pkl"
        meta_vocab_size = None
        if meta_path.exists():
            # Standard Library
            import pickle

            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            meta_vocab_size = meta["vocab_size"]
            logger.info(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        if self.config.init_from == "scratch":
            # Fresh random init
            vocab_size = meta_vocab_size if meta_vocab_size is not None else self.config.vocab_size
            if meta_vocab_size is None:
                logger.info(f"defaulting to vocab_size of {vocab_size}")

            self.model = gpt.GPT(
                n_layer=self.config.n_layer,
                n_head=self.config.n_head,
                n_embd=self.config.n_embd,
                vocab_size=vocab_size,
                block_size=self.config.block_size,
                dropout=self.config.dropout,
                bias=self.config.bias,
            )

        elif self.config.init_from == "resume":
            # Resume from checkpoint - handled in maybe_resume_training
            pass

        elif self.config.init_from.startswith("gpt2"):
            self.model = gpt.GPT.from_pretrained(self.config.init_from)

            # Might need to crop block size
            if self.config.block_size < self.model.config.block_size:
                self.model.crop_block_size(self.config.block_size)

        if self.model is not None:
            self.model.to(self.device)

    def init_optimizer(self):
        """Initialize AdamW optimizer with weight decay."""
        if self.model is None:
            return  # Will be initialized after loading checkpoint

        self.optimizer = self.model.configure_optimizers(
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            device_type=self.config.device_type,
        )

    def init_dataloader(self):
        """Initialize data loader for train and validation."""
        self.dataloader = dataloader.PretrainDataLoader(
            data_dir=pathlib.Path(self.config.data_dir) / self.config.dataset,
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            device=self.device,
            device_type=self.config.device_type,
        )

    def init_lr_scheduler(self):
        """Initialize learning rate scheduler."""
        self.lr_scheduler = lr_scheduler.CosineScheduler(
            learning_rate=self.config.learning_rate,
            warmup_iters=self.config.warmup_iters,
            lr_decay_iters=self.config.lr_decay_iters,
            min_lr=self.config.min_lr,
        )

    def init_checkpointer(self):
        """Initialize checkpointing utility."""
        self.checkpointer = checkpointer.Checkpointer(
            out_dir=self.config.out_dir, config=self.config
        )

    def init_scaler(self):
        """Initialize gradient scaler for mixed precision."""
        # Only use scaler for float16 training
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == "float16"))

    def maybe_resume_training(self):
        """Resume from checkpoint if requested."""
        if self.config.init_from == "resume":
            checkpoint = self.checkpointer.load_checkpoint(self.device)

            # Get model args from checkpoint
            checkpoint_model_args = checkpoint["model_args"]

            # Force these config attributes to be equal otherwise we can't resume training
            for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
                setattr(self.config, k, checkpoint_model_args[k])

            # Create model with checkpoint config
            self.model = gpt.GPT(
                n_layer=checkpoint_model_args["n_layer"],
                n_head=checkpoint_model_args["n_head"],
                n_embd=checkpoint_model_args["n_embd"],
                vocab_size=checkpoint_model_args["vocab_size"],
                block_size=checkpoint_model_args["block_size"],
                dropout=checkpoint_model_args.get("dropout", 0.0),
                bias=checkpoint_model_args["bias"],
            )

            # Load state dict with unwanted prefix handling
            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)

            # Initialize optimizer and load state
            self.init_optimizer()
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            # Restore training state
            self.iter_num = checkpoint["iter_num"]
            self.best_val_loss = checkpoint["best_val_loss"]

            logger.info(f"Resumed from iteration {self.iter_num}")

    def maybe_compile_model(self):
        """Compile model with PyTorch 2.0 if requested."""
        if self.config.compile:
            logger.info("Compiling model... (takes ~1 minute)")
            self.model = torch.compile(self.model)

    def maybe_wrap_ddp(self):
        """Wrap model in DDP if using distributed training."""
        if self.ddp_world_size > 1:
            self.model = parallel.DistributedDataParallel(
                self.model, device_ids=[self.ddp_local_rank]
            )

    def training_loop(self):
        """
        Main training loop with clear structure.
        Each iteration follows the same pattern.
        """
        # Get first batch
        X, Y = self.dataloader.get_batch("train")

        t0 = time.time()

        logger.info(f"Starting training from iteration {self.iter_num}")
        logger.info(f"Tokens per iteration: {self.get_tokens_per_iter():,}")

        running_mfu = -1.0
        local_iter_num = 0

        while self.iter_num < self.config.max_iters:
            self.update_learning_rate()

            if self.should_evaluate():
                self.evaluate_and_checkpoint(running_mfu)

            X, Y = self.training_step(X, Y)

            if self.should_log():
                running_mfu = self.log_progress(t0, running_mfu, local_iter_num)
                t0 = time.time()

            self.iter_num += 1
            local_iter_num += 1

    def update_learning_rate(self):
        """Update learning rate based on schedule."""
        if self.config.decay_lr:
            lr = self.lr_scheduler.get_lr(self.iter_num)
        else:
            lr = self.config.learning_rate

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def should_evaluate(self):
        """Check if we should run evaluation."""
        return self.iter_num % self.config.eval_interval == 0 and distribution.is_main_process()

    def evaluate_and_checkpoint(self, running_mfu):
        """Run evaluation and save checkpoint if improved."""
        losses = self.estimate_loss()
        logger.info(
            f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        if self.config.wandb_log:
            lr = self.optimizer.param_groups[0]["lr"]
            self.log_to_wandb(losses, lr, running_mfu)

        if losses["val"] < self.best_val_loss or self.config.always_save_checkpoint:
            if losses["val"] < self.best_val_loss:
                self.best_val_loss = losses["val"]

            if self.iter_num > 0:
                self.save_checkpoint()

    def training_step(self, X, Y):
        """
        Single training step with gradient accumulation.
        Clear sequence of operations for educational value.
        """
        # Accumulate gradients over micro-batches
        for micro_step in range(self.config.gradient_accumulation_steps):
            # Sync gradients only on last micro-step (DDP optimization)
            if self.ddp_world_size > 1:
                self.model.require_backward_grad_sync = (
                    micro_step == self.config.gradient_accumulation_steps - 1
                )

            # Forward pass with autocast
            with self.ctx:
                logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"]
                loss: jaxtyping.Float[torch.Tensor, ""]

                logits, loss = self.model(X, Y)
                # Scale loss by gradient accumulation steps
                loss = loss / self.config.gradient_accumulation_steps

            # Immediately async prefetch next batch while model is doing forward pass on GPU
            X, Y = self.dataloader.get_batch("train")

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.config.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Flush gradients as soon as we can, no need for this memory anymore
        self.optimizer.zero_grad(set_to_none=True)

        # Store loss for logging (unscaled)
        self.loss = loss.item() * self.config.gradient_accumulation_steps

        # Return next batch for use in training loop
        return X, Y

    def should_log(self):
        """Check if we should log progress."""
        return self.iter_num % self.config.log_interval == 0 and distribution.is_main_process()

    def log_progress(self, t0, running_mfu, local_iter_num):
        """Log training progress."""
        # Calculate time
        t1 = time.time()
        dt = t1 - t0

        # Get loss (already computed in training_step)
        lossf = self.loss

        # Estimate model flops utilization only after warmup
        if local_iter_num >= 5:
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            mfu = raw_model.estimate_mfu(
                self.config.batch_size * self.config.gradient_accumulation_steps, dt
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        logger.info(
            f"iter {self.iter_num}: loss {lossf:.4f}, "
            f"time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
        )

        return running_mfu

    @torch.no_grad()
    def estimate_loss(self):
        """Estimate loss over multiple batches for train/val."""
        out = {}
        self.model.eval()

        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.dataloader.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()

        self.model.train()
        return out

    def save_checkpoint(self):
        """Save training checkpoint."""
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        # Get model args for checkpoint
        model_args = {
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "n_embd": self.config.n_embd,
            "block_size": self.config.block_size,
            "bias": self.config.bias,
            "vocab_size": raw_model.config.vocab_size,  # Get actual vocab size from model
            "dropout": self.config.dropout,
        }

        self.checkpointer.save_checkpoint(
            model=raw_model,
            optimizer=self.optimizer,
            iter_num=self.iter_num,
            best_val_loss=self.best_val_loss,
            model_args=model_args,
            config=vars(self.config),  # Convert dataclass to dict
        )

    def log_to_wandb(self, losses, lr, running_mfu):
        """Log metrics to Weights & Biases."""
        wandb.log(
            {
                "iter": self.iter_num,
                "train/loss": losses["train"],
                "val/loss": losses["val"],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            }
        )

    def get_tokens_per_iter(self):
        """Calculate total tokens processed per iteration."""
        return (
            self.config.batch_size
            * self.config.block_size
            * self.config.gradient_accumulation_steps
            * self.ddp_world_size
        )

    def setup(self):
        """
        Setup all components needed for training.
        Separated from train() for clarity and flexibility.
        """
        # Environment setup
        self.setup_distributed_training()
        self.setup_device()
        self.setup_directories()
        self.set_seed()

        # Initialize all components
        self.init_model()
        self.init_optimizer()
        self.init_dataloader()
        self.init_lr_scheduler()
        self.init_checkpointer()
        self.init_scaler()

        # Resume from checkpoint if requested
        self.maybe_resume_training()

        # Wrap and/or compile model if needed
        self.maybe_compile_model()
        self.maybe_wrap_ddp()

        # Setup wandb logging if enabled
        if self.config.wandb_log and distribution.is_main_process():
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )

        logger.info("Setup complete. Ready to train.")

    def train(self):
        """
        Run the main training loop.
        Assumes setup() has been called.
        """
        if self.model is None:
            raise RuntimeError("Must call setup() before train()")

        self.training_loop()
        distribution.cleanup_ddp()
