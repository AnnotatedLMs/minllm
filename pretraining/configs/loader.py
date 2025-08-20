# Standard Library
import pathlib
import typing

# Third Party
import yaml

# Project
from pretraining.configs import core
from pretraining.configs.training import trainer_configs


def load_training_config(
    config_path: typing.Union[str, pathlib.Path],
    model_config_class: typing.Type[core.ModelConfigT],
) -> core.TrainerConfig[core.ModelConfigT]:
    """Load a training configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file
        model_config_class: The model config class to use (e.g., GPT2Config, LlamaConfig)

    Returns:
        Parsed and validated TrainerConfig with proper generic typing

    Example:
        from pretraining.configs.model.architectures import gpt

        config = load_training_config(
            "configs/gpt2.yaml",
            gpt.GPT2Config
        )
    """
    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Use architecture-specific from_yaml method
    model_config = model_config_class.from_yaml(str(config_path))

    # Load YAML again for training config
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Parse training config using model_validate for proper type conversion
    training_config = trainer_configs.TrainingLoopConfig.model_validate(config_dict["training"])

    # Create the complete trainer config with proper generic type
    trainer_config = core.TrainerConfig[model_config_class](
        llm=model_config, training=training_config
    )

    return trainer_config
