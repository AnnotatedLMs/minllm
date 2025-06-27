# Standard Library
import dataclasses
import pathlib
import typing

# Third Party
import yaml


@dataclasses.dataclass
class TrainerConfig:
    """Unified configuration for all pretraining components."""

    # Model Architecture
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int
    block_size: int
    dropout: float
    bias: bool

    # Training
    batch_size: int
    gradient_accumulation_steps: int
    max_iters: int
    eval_interval: int
    eval_iters: int
    log_interval: int
    always_save_checkpoint: bool
    init_from: str  # 'scratch', 'resume', or 'gpt2*'

    # Optimizer
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float

    # Learning Rate Schedule
    decay_lr: bool
    warmup_iters: int
    lr_decay_iters: int
    min_lr: float

    # Data
    dataset: str
    data_dir: str

    # System
    device: str
    dtype: str
    compile: bool
    backend: str

    # Logging
    out_dir: str
    wandb_log: bool
    wandb_project: str
    wandb_run_name: str

    # Reproducibility
    seed: int

    def __post_init__(self):
        """Validate config values after initialization."""
        # Validate dtype
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"Invalid dtype: {self.dtype}. Must be float32, float16, or bfloat16")

        # Validate init_from
        if not (
            self.init_from == "scratch"
            or self.init_from == "resume"
            or self.init_from.startswith("gpt2")
        ):
            raise ValueError(
                f"Invalid init_from: {self.init_from}. Must be 'scratch', 'resume', or 'gpt2*'"
            )

        # Validate numeric ranges
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.min_lr > self.learning_rate:
            raise ValueError(
                f"min_lr ({self.min_lr}) must be <= learning_rate ({self.learning_rate})"
            )
        if self.warmup_iters < 0:
            raise ValueError(f"warmup_iters must be non-negative, got {self.warmup_iters}")
        if self.lr_decay_iters < self.warmup_iters:
            raise ValueError(
                f"lr_decay_iters ({self.lr_decay_iters}) must be >= warmup_iters ({self.warmup_iters})"
            )

    @property
    def tokens_per_iter(self) -> int:
        """Calculate tokens processed per iteration."""
        # This will be adjusted by DDP world size in the trainer
        return self.batch_size * self.block_size * self.gradient_accumulation_steps

    @property
    def device_type(self) -> str:
        """Extract device type for autocast."""
        return "cuda" if "cuda" in self.device else "cpu"


def load_config(config_path: typing.Union[str, pathlib.Path]) -> TrainerConfig:
    """Load configuration from YAML file."""
    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    try:
        return TrainerConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid config file: {e}") from e
