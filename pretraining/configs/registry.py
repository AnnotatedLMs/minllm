# Standard Library
import pathlib
import typing

# Third Party
import yaml

# Project
from pretraining.configs import core
from pretraining.configs.llm import enums

# Local
from pretraining.configs.parsers import model_parser
from pretraining.configs.parsers import training_parser


class ConfigRegistry:
    """
    Central registry for configuration parsing.

    Provides a unified interface for loading and parsing YAML configurations
    into structured config objects.
    """

    def __init__(
        self,
        model_parser: model_parser.ModelConfigParser,
        training_parser: training_parser.TrainingConfigParser,
    ):
        self.model_parser = model_parser
        self.training_parser = training_parser

    def load_yaml(self, yaml_path: typing.Union[str, pathlib.Path]) -> typing.Dict[str, typing.Any]:
        """
        Load YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Dictionary containing raw YAML data

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        yaml_path = pathlib.Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    def parse_config(self, yaml_path: typing.Union[str, pathlib.Path]) -> core.TrainerConfig:
        """
        Main entry point: Load and parse complete configuration.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Complete TrainerConfig ready for training
        """
        config_dict = self.load_yaml(yaml_path)

        # Parse model components
        if "model" not in config_dict:
            raise ValueError("Configuration must include 'model' section")
        model_dict = config_dict["model"]

        # Extract architecture type
        if "architecture" not in model_dict:
            raise ValueError("Model configuration must include 'architecture'")
        architecture_str = model_dict["architecture"]
        try:
            architecture = enums.Architecture(architecture_str)
        except ValueError:
            raise ValueError(
                f"Unknown architecture: {architecture_str}. Must be one of: {[a.value for a in enums.Architecture]}"
            )

        # Parse token embedding config
        if "token_embedding" not in model_dict:
            raise ValueError("Model configuration must include 'token_embedding' section")
        token_embedding_config = self.model_parser.parse_token_embedding_config(
            model_dict["token_embedding"]
        )

        # Parse position embedding config (optional)
        position_embedding_config = None
        if "position_embedding" in model_dict:
            position_embedding_config = self.model_parser.parse_position_embedding_config(
                model_dict["position_embedding"]
            )

        # Parse transformer backbone
        if "transformer_backbone" not in model_dict:
            raise ValueError("Model configuration must include 'transformer_backbone' section")
        # TODO: Consider if we just want to parse the transformer config directly (rather than each sub-component)
        transformer_dict = model_dict["transformer_backbone"]

        if "attention" not in transformer_dict:
            raise ValueError("Transformer configuration must include 'attention' section")
        attention_config = self.model_parser.parse_attention_config(transformer_dict["attention"])

        if "normalization" not in transformer_dict:
            raise ValueError("Transformer configuration must include 'normalization' section")
        normalization_config = self.model_parser.parse_normalization_config(
            transformer_dict["normalization"]
        )
        rope_config = None
        if "rope" in transformer_dict:
            rope_config = self.model_parser.parse_rope_config(transformer_dict["rope"])

        ffn_config = None
        moe_config = None
        if "moe" in transformer_dict:
            moe_config = self.model_parser.parse_moe_config(transformer_dict["moe"])
        else:
            # Default to FFN if no MoE
            if "ffn" not in transformer_dict:
                raise ValueError(
                    "Transformer configuration must include either 'moe' or 'ffn' section"
                )
            ffn_config = self.model_parser.parse_ffn_config(transformer_dict["ffn"])

        # Create transformer config
        transformer_config = self.model_parser.create_transformer_config(
            transformer_dict=transformer_dict,
            attention_config=attention_config,
            ffn_config=ffn_config,
            normalization_config=normalization_config,
            rope_config=rope_config,
            moe_config=moe_config,
        )

        # Parse output head config
        if "output_head" not in model_dict:
            raise ValueError("Model configuration must include 'output_head' section")
        output_head_config = self.model_parser.parse_output_head_config(model_dict["output_head"])

        # Parse MTP config (optional)
        mtp_config = None
        if "mtp" in model_dict:
            mtp_config = self.model_parser.parse_mtp_config(model_dict["mtp"])

        # Parse weight initialization config (required)
        if "weight_init" not in model_dict:
            raise ValueError("Model configuration must include 'weight_init' section")
        weight_init_config = self.model_parser.parse_weight_init_config(model_dict["weight_init"])

        # Create complete LLM config
        llm_config = self.model_parser.create_llm_config(
            token_embedding_config=token_embedding_config,
            transformer_config=transformer_config,
            output_head_config=output_head_config,
            weight_init_config=weight_init_config,
            position_embedding_config=position_embedding_config,
            mtp_config=mtp_config,
            architecture=architecture,
        )

        # Parse training components
        # Note: Adjusting to match the expected structure from trainer_configs.py
        if "training" not in config_dict:
            raise ValueError("Configuration must include 'training' section")
        training_dict = config_dict["training"]

        # Parse batch config
        if "batch_size" not in training_dict:
            raise ValueError("Training configuration must include 'batch_size'")
        if "gradient_accumulation_steps" not in training_dict:
            raise ValueError("Training configuration must include 'gradient_accumulation_steps'")

        batch_config = self.training_parser.parse_batch_config(
            {
                "batch_size": training_dict["batch_size"],
                "gradient_accumulation_steps": training_dict["gradient_accumulation_steps"],
                "sequence_length": transformer_config.block_size,
            }
        )

        # Parse optimizer config
        if "optimizer" not in config_dict:
            raise ValueError("Configuration must include 'optimizer' section")
        optimizer_config = self.training_parser.parse_optimizer_config(config_dict["optimizer"])

        # Parse LR schedule config
        if "lr_schedule" not in config_dict:
            raise ValueError("Configuration must include 'lr_schedule' section")
        lr_schedule_config = self.training_parser.parse_lr_schedule_config(
            config_dict["lr_schedule"]
        )

        # Parse evaluation config
        evaluation_config = self.training_parser.parse_evaluator_config(training_dict)

        # Parse checkpoint config
        if "checkpointing" not in config_dict:
            raise ValueError("Configuration must include 'checkpointing' section")
        checkpoint_config = self.training_parser.parse_checkpointer_config(
            config_dict["checkpointing"]
        )

        # Parse data config
        if "data" not in config_dict:
            raise ValueError("Configuration must include 'data' section")
        data_config = self.training_parser.parse_data_config(config_dict["data"])

        # Parse system configs
        if "system" not in config_dict:
            raise ValueError("Configuration must include 'system' section")
        system_dict = config_dict["system"]
        device_config = self.training_parser.parse_device_config(system_dict)
        torch_compilation_config = self.training_parser.parse_torch_compilation_config(system_dict)
        distribution_config = self.training_parser.parse_distributed_config(system_dict)

        # Parse logging config
        if "logging" not in config_dict:
            raise ValueError("Configuration must include 'logging' section")
        wandb_logging_config = self.training_parser.parse_wandb_config(config_dict["logging"])

        # Parse optional training components based on model config
        moe_training_config = None
        if moe_config is not None:
            moe_training_config = self.training_parser.parse_moe_training_config(
                training_dict.get("moe_training", {})
            )

        mtp_training_config = None
        if mtp_config is not None:
            mtp_training_config = self.training_parser.parse_mtp_training_config(
                training_dict.get("mtp_training", {})
            )

        # Create complete training loop config
        training_loop_config = self.training_parser.create_training_loop_config(
            max_iters=training_dict["max_iters"],
            log_interval=training_dict["log_interval"],
            batch_config=batch_config,
            optimizer_config=optimizer_config,
            lr_schedule_config=lr_schedule_config,
            evaluation_config=evaluation_config,
            checkpoint_config=checkpoint_config,
            data_config=data_config,
            device_config=device_config,
            torch_compilation_config=torch_compilation_config,
            distribution_config=distribution_config,
            wandb_logging_config=wandb_logging_config,
            moe_training_config=moe_training_config,
            mtp_training_config=mtp_training_config,
            seed=training_dict.get("seed", 1337),  # Default seed if not specified
        )

        return core.TrainerConfig(
            llm=llm_config,
            training=training_loop_config,
        )


# Convenience function for backward compatibility
def load_training_config(yaml_path: typing.Union[str, pathlib.Path]) -> core.TrainerConfig:
    """
    Load a complete training configuration from a YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Complete TrainerConfig ready for training
    """
    registry = ConfigRegistry()
    return registry.parse_config(yaml_path)
