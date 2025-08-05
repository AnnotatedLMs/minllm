# Standard Library
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Union


@dataclass
class SFTArguments:
    """Arguments for supervised fine-tuning."""

    # Model configuration
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version to use."},
    )
    use_flash_attn: bool = field(
        default=True, metadata={"help": "Whether to use flash attention in the model training"}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={"help": "Create model as empty shell, then materialize parameters."},
    )

    # Dataset configuration
    dataset_mixer_list: List[str] = field(
        default_factory=list, metadata={"help": "List of datasets to sample from."}
    )
    dataset_mixer_list_splits: List[str] = field(
        default_factory=lambda: ["train"],
        metadata={"help": "The dataset splits to use for training"},
    )
    dataset_transform_fn: List[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
        metadata={"help": "Transform functions to apply to dataset."},
    )
    max_seq_length: Optional[int] = field(
        default=None, metadata={"help": "Maximum total input sequence length after tokenization."}
    )

    # Training configuration
    output_dir: str = field(
        default="output/",
        metadata={"help": "Output directory for model checkpoints."},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate."}
    )
    learning_rate: float = field(
        default=2e-5, metadata={"help": "Initial learning rate for AdamW."}
    )
    num_train_epochs: int = field(default=2, metadata={"help": "Total number of training epochs."})
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Override number of training steps."},
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={"help": "Learning rate scheduler type."},
    )

    # Optimization configuration
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW."})
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. -1 means no clipping."},
    )
    fused_optimizer: bool = field(default=True, metadata={"help": "Whether to use fused AdamW."})
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Turn on gradient checkpointing."}
    )

    # Loss configuration
    reduce_loss: str = field(
        default="mean",
        metadata={"help": "How to reduce loss over tokens. Options: 'mean' or 'sum'."},
    )

    # LoRA configuration
    use_lora: bool = field(
        default=False,
        metadata={"help": "Use LoRA (low-rank adaptation) training."},
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Use QLoRA (quantized LoRA) training."},
    )
    lora_rank: int = field(default=64, metadata={"help": "The rank of LoRA."})
    lora_alpha: float = field(default=16, metadata={"help": "The alpha parameter of LoRA."})
    lora_dropout: float = field(default=0.1, metadata={"help": "The dropout rate of LoRA modules."})

    # Checkpointing configuration
    checkpointing_steps: Optional[str] = field(
        default=None, metadata={"help": "Save states every n steps, or 'epoch' for each epoch."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Resume training from checkpoint folder."}
    )

    # Logging configuration
    logging_steps: Optional[int] = field(default=None, metadata={"help": "Log every n steps."})
    report_to: Union[str, List[str]] = field(
        default="all", metadata={"help": "Integration(s) to report to."}
    )

    # Memory optimization
    packing: bool = field(
        default=False,
        metadata={"help": "Use packing/padding-free collation."},
    )

    # Other configuration
    seed: int = field(default=42, metadata={"help": "Random seed for initialization."})

    def __post_init__(self):
        """Validate arguments after initialization."""
        if self.reduce_loss not in ["mean", "sum"]:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")

        if self.use_qlora and self.use_lora:
            raise ValueError("Cannot use both LoRA and QLoRA")

        if self.use_qlora and self.fused_optimizer:
            raise ValueError("Cannot use fused optimizer with QLoRA")
