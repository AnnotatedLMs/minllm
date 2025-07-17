# Standard Library
import typing

# Project
# Local
from pretraining.configs.training import architecture_training_configs
from pretraining.configs.training import batch_configs
from pretraining.configs.training import checkpointer_configs
from pretraining.configs.training import data_configs
from pretraining.configs.training import evaluator_configs
from pretraining.configs.training import lr_configs
from pretraining.configs.training import optimizer_configs
from pretraining.configs.training import system_configs
from pretraining.configs.training import trainer_configs
from pretraining.configs.training import wandb_configs


class TrainingConfigParser:
    """Parses training-related configuration components from dictionaries."""

    def parse_optimizer_config(
        self, opt_dict: typing.Dict[str, typing.Any]
    ) -> optimizer_configs.OptimizerConfig:
        """Parse optimizer configuration."""
        return optimizer_configs.OptimizerConfig(
            optimizer_type=opt_dict["optimizer_type"],
            learning_rate=opt_dict["learning_rate"],
            weight_decay=opt_dict["weight_decay"],
            beta1=opt_dict["beta1"],
            beta2=opt_dict["beta2"],
            grad_clip=opt_dict["grad_clip"],
            eps=opt_dict["eps"],
            parameter_grouping=opt_dict["parameter_grouping"],
            no_decay_patterns=opt_dict.get("no_decay_patterns"),  # Optional
        )

    def parse_lr_schedule_config(
        self, lr_dict: typing.Dict[str, typing.Any]
    ) -> lr_configs.LearningRateScheduleConfig:
        """Parse learning rate schedule configuration."""
        config = lr_configs.LearningRateScheduleConfig(
            schedule_type=lr_dict["schedule_type"],
            warmup_iters=lr_dict["warmup_iters"],
            lr_decay_iters=lr_dict["lr_decay_iters"],
            min_lr=lr_dict["min_lr"],
            num_cycles=lr_dict.get("num_cycles"),  # Optional
            decay_lr=lr_dict.get("decay_lr"),  # Optional
        )

        # Handle multiphase schedule
        if config.schedule_type == "multiphase":
            config.phase_steps = lr_dict.get("phase_steps")
            config.phase_names = lr_dict.get("phase_names")
            if config.phase_steps is None:
                raise ValueError("lr_schedule.phase_steps required for multiphase schedule")

        return config

    def parse_batch_config(
        self, batch_dict: typing.Dict[str, typing.Any]
    ) -> batch_configs.BatchConfig:
        """Parse batch configuration."""
        return batch_configs.BatchConfig(
            batch_size=batch_dict["batch_size"],
            gradient_accumulation_steps=batch_dict["gradient_accumulation_steps"],
            sequence_length=batch_dict["sequence_length"],
        )

    def parse_data_config(self, data_dict: typing.Dict[str, typing.Any]) -> data_configs.DataConfig:
        """Parse data configuration."""
        return data_configs.DataConfig(
            dataset=data_dict["dataset"],
            data_dir=data_dict["data_dir"],
            num_workers=data_dict["num_workers"],
            pin_memory=data_dict["pin_memory"],
        )

    def parse_device_config(
        self, device_dict: typing.Dict[str, typing.Any]
    ) -> system_configs.DeviceConfig:
        """Parse device configuration."""
        return system_configs.DeviceConfig(
            device=device_dict["device"],
            dtype=device_dict.get(
                "torch_dtype", device_dict.get("dtype", "float32")
            ),  # Handle both field names
        )

    def parse_torch_compilation_config(
        self, compile_dict: typing.Dict[str, typing.Any]
    ) -> system_configs.TorchCompilationConfig:
        """Parse torch compilation configuration."""
        return system_configs.TorchCompilationConfig(
            compile=compile_dict["compile"],
            compile_mode=compile_dict.get("compile_mode"),  # Optional
        )

    def parse_distributed_config(
        self, dist_dict: typing.Dict[str, typing.Any]
    ) -> system_configs.DistributedConfig:
        """Parse distributed training configuration."""
        return system_configs.DistributedConfig(
            backend=dist_dict.get("backend", "nccl"),  # Default to nccl
            ddp_bucket_cap_mb=dist_dict.get("ddp_bucket_cap_mb"),  # Optional
            find_unused_parameters=dist_dict.get(
                "find_unused_parameters", False
            ),  # Default to False
        )

    def parse_wandb_config(
        self, wandb_dict: typing.Dict[str, typing.Any]
    ) -> wandb_configs.WandbConfig:
        """Parse Weights & Biases logging configuration."""
        return wandb_configs.WandbConfig(
            enabled=wandb_dict["enabled"],
            project=wandb_dict["project"],
            run_name=wandb_dict.get("run_name"),  # Optional
            tags=wandb_dict.get("tags"),  # Optional
            log_model=wandb_dict["log_model"],
        )

    def parse_checkpointer_config(
        self, ckpt_dict: typing.Dict[str, typing.Any]
    ) -> checkpointer_configs.CheckpointerConfig:
        """Parse checkpointing configuration."""
        return checkpointer_configs.CheckpointerConfig(
            save_dir=ckpt_dict["save_dir"],
            save_interval=ckpt_dict["save_interval"],
            save_best=ckpt_dict["save_best"],
            keep_last_n=ckpt_dict["keep_last_n"],
            resume_from=ckpt_dict.get("resume_from"),  # Optional
        )

    def parse_evaluator_config(
        self, eval_dict: typing.Dict[str, typing.Any]
    ) -> evaluator_configs.EvaluatorConfig:
        """Parse evaluation configuration."""
        return evaluator_configs.EvaluatorConfig(
            eval_interval=eval_dict["eval_interval"],
            eval_iters=eval_dict["eval_iters"],
            eval_only=eval_dict.get("eval_only", False),  # Optional, default to training mode
        )

    def parse_moe_training_config(
        self, moe_dict: typing.Dict[str, typing.Any]
    ) -> architecture_training_configs.MoETrainingConfig:
        """Parse MoE training configuration."""
        return architecture_training_configs.MoETrainingConfig(
            aux_loss_weight=moe_dict["aux_loss_weight"],
            capacity_factor=moe_dict["capacity_factor"],
            drop_tokens=moe_dict["drop_tokens"],
            z_loss_weight=moe_dict["z_loss_weight"],
        )

    def parse_mtp_training_config(
        self, mtp_dict: typing.Dict[str, typing.Any]
    ) -> architecture_training_configs.MultiTokenPredictionTrainingConfig:
        """Parse multi-token prediction training configuration."""
        return architecture_training_configs.MultiTokenPredictionTrainingConfig(
            mtp_loss_weight=mtp_dict["mtp_loss_weight"],
        )

    def create_training_loop_config(
        self,
        max_iters: int,
        log_interval: int,
        batch_config: batch_configs.BatchConfig,
        optimizer_config: optimizer_configs.OptimizerConfig,
        lr_schedule_config: lr_configs.LearningRateScheduleConfig,
        evaluation_config: evaluator_configs.EvaluatorConfig,
        checkpoint_config: checkpointer_configs.CheckpointerConfig,
        data_config: data_configs.DataConfig,
        device_config: system_configs.DeviceConfig,
        torch_compilation_config: system_configs.TorchCompilationConfig,
        distribution_config: system_configs.DistributedConfig,
        wandb_logging_config: wandb_configs.WandbConfig,
        moe_training_config: typing.Optional[
            architecture_training_configs.MoETrainingConfig
        ] = None,
        mtp_training_config: typing.Optional[
            architecture_training_configs.MultiTokenPredictionTrainingConfig
        ] = None,
        seed: int = 1337,
    ) -> trainer_configs.TrainingLoopConfig:
        """Create complete training loop configuration from parsed components."""
        return trainer_configs.TrainingLoopConfig(
            max_iters=max_iters,
            log_interval=log_interval,
            batch=batch_config,
            optimizer=optimizer_config,
            lr_schedule=lr_schedule_config,
            evaluation=evaluation_config,
            checkpoint=checkpoint_config,
            data=data_config,
            device=device_config,
            torch_compilation=torch_compilation_config,
            distribution=distribution_config,
            wandb_logging=wandb_logging_config,
            moe_training=moe_training_config,
            mtp_training=mtp_training_config,
            seed=seed,
        )
