# Standard Library
import typing

# Third Party
import pydantic

# Project
from pretraining.configs import base
from pretraining.configs.training import architecture_training_configs
from pretraining.configs.training import batch_configs
from pretraining.configs.training import checkpointer_configs
from pretraining.configs.training import data_configs
from pretraining.configs.training import evaluator_configs
from pretraining.configs.training import execution_configs
from pretraining.configs.training import lr_configs
from pretraining.configs.training import optimizer_configs
from pretraining.configs.training import system_configs
from pretraining.configs.training import wandb_configs


class TrainingLoopConfig(base.BaseConfig):
    """
    Configuration for training loop parameters.
    """

    # 1. Data configuration (what we're training on)
    data: data_configs.DataConfig

    # 2. Batch processing (how we feed data to the model)
    batch: batch_configs.BatchConfig

    # 3. Optimization settings (how we update the model)
    optimizer: optimizer_configs.OptimizerConfig
    lr_schedule: lr_configs.LearningRateScheduleConfig

    # 4. Training loop control (when to evaluate, checkpoint, and stop)
    max_iters: int = pydantic.Field(gt=0)  # Total number of training iterations
    evaluation: evaluator_configs.EvaluatorConfig
    checkpoint: checkpointer_configs.CheckpointerConfig

    # 5. System configuration (hardware and execution)
    torch_compilation: system_configs.TorchCompilationConfig
    execution: execution_configs.ExecutionConfig

    # 6. Monitoring and logging
    log_interval: int = pydantic.Field(gt=0)  # How often to log training metrics
    wandb_logging: wandb_configs.WandbConfig

    # 7. Reproducibility
    seed: int = pydantic.Field(ge=0)  # Random seed for reproducibility

    # 8. Architecture-specific training settings (optional)
    moe_training: typing.Optional[architecture_training_configs.MoETrainingConfig] = (
        None  # For MoE models
    )
    mtp_training: typing.Optional[
        architecture_training_configs.MultiTokenPredictionTrainingConfig
    ] = None  # For multi-token prediction
