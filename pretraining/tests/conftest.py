"""
Shared test fixtures for MinLLM tests.

This file is automatically loaded by pytest and makes fixtures
available to all test files without explicit imports.
"""

# Standard Library
import pathlib

# Third Party
import pytest

# Project
from pretraining.common.models.architectures import deepseek3
from pretraining.common.models.architectures import gpt2
from pretraining.common.models.architectures import llama3
from pretraining.configs import loader
from pretraining.configs.model.architectures import deepseek
from pretraining.configs.model.architectures import gpt
from pretraining.configs.model.architectures import llama


@pytest.fixture(scope="session")
def project_root() -> pathlib.Path:
    """Get the project root directory."""
    return pathlib.Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def debug_configs_dir(project_root: pathlib.Path) -> pathlib.Path:
    """Path to debug configs directory."""
    return project_root / "pretraining" / "configs" / "examples" / "debug"


@pytest.fixture(scope="session")
def test_data_dir(project_root: pathlib.Path) -> pathlib.Path:
    """Path to test data directory."""
    return project_root / "pretraining" / "tests" / "fixtures" / "data"


# Model fixtures - these can be expensive to create so we cache them at module level
@pytest.fixture(scope="module")
def gpt2_debug_model(debug_configs_dir: pathlib.Path) -> gpt2.GPT2:
    """Create a small GPT-2 model for testing."""
    config_path = debug_configs_dir / "gpt2_debug_cpu.yaml"
    config = loader.load_training_config(config_path, gpt.GPT2Config)
    return gpt2.GPT2.from_config(config.llm)


@pytest.fixture(scope="module")
def llama_debug_model(debug_configs_dir: pathlib.Path) -> llama3.Llama3:
    """Create a small Llama model for testing."""
    config_path = debug_configs_dir / "llama3_debug_cpu.yaml"
    config = loader.load_training_config(config_path, llama.Llama3Config)
    return llama3.Llama3.from_config(config.llm)


@pytest.fixture(scope="module")
def deepseek_debug_model(debug_configs_dir: pathlib.Path) -> deepseek3.DeepSeek3:
    """Create a small DeepSeek model for testing."""
    config_path = debug_configs_dir / "deepseek3_debug_cpu.yaml"
    config = loader.load_training_config(config_path, deepseek.DeepSeek3Config)
    return deepseek3.DeepSeek3.from_config(config.llm)


# Config fixtures - useful for tests that need configs but not models
@pytest.fixture
def gpt2_debug_config(debug_configs_dir: pathlib.Path) -> gpt.GPT2Config:
    """Load GPT-2 debug config."""
    config_path = debug_configs_dir / "gpt2_debug_cpu.yaml"
    return loader.load_training_config(config_path, gpt.GPT2Config).llm


@pytest.fixture
def llama_debug_config(debug_configs_dir: pathlib.Path) -> llama.Llama3Config:
    """Load Llama debug config."""
    config_path = debug_configs_dir / "llama3_debug_cpu.yaml"
    return loader.load_training_config(config_path, llama.Llama3Config).llm


@pytest.fixture
def deepseek_debug_config(debug_configs_dir: pathlib.Path) -> deepseek.DeepSeek3Config:
    """Load DeepSeek debug config."""
    config_path = debug_configs_dir / "deepseek3_debug_cpu.yaml"
    return loader.load_training_config(config_path, deepseek.DeepSeek3Config).llm
