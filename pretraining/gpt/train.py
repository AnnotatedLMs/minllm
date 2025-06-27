# Standard Library
import argparse
import pathlib

# Third Party
import yaml

# Project
from pretraining.gpt.trainers import pretrain
from pretraining.gpt.utils import config


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model from a configuration file")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    return parser.parse_args()


def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    path = pathlib.Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    return config_dict


def main():
    args = parse_args()

    print(f"Loading config from: {args.config}")
    config_dict = load_yaml_config(args.config)

    trainer_config = config.TrainerConfig(**config_dict)

    print("Initializing trainer...")
    gpt_trainer = pretrain.Trainer(trainer_config)

    print("Setting up training components...")
    gpt_trainer.setup()

    print(f"Starting training for {trainer_config.max_iters} iterations...")
    gpt_trainer.train()

    print("Training completed!")


if __name__ == "__main__":
    main()
