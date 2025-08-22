# MinLLM Pretraining

This directory contains the pretraining infrastructure for MinLLM, supporting GPT-2, Llama 3, and DeepSeek 3 architectures.

## Quick Start

### Training Scripts

```bash
# Train GPT-2
uv run python pretraining/scripts/train_gpt2.py

# Train Llama 3
uv run python pretraining/scripts/train_llama3.py

# Train DeepSeek 3
uv run python pretraining/scripts/train_deepseek3.py
```

Each script uses debug configurations by default for quick testing. To use custom configs, modify the config paths in the scripts.

### Running Tests

```bash
# Run all tests
uv run pytest pretraining/tests/ -v

# Run specific test categories
uv run pytest pretraining/tests/unit/ -v          # Unit tests only
uv run pytest pretraining/tests/integration/ -v   # Integration tests only

# Run tests for specific components
uv run pytest pretraining/tests/unit/patterns/test_attention.py -v
uv run pytest pretraining/tests/unit/patterns/test_moe.py -v
uv run pytest pretraining/tests/unit/configs/test_config_parsing.py -v
```

## Configuration

All models are configured through YAML files. Example configurations are in `pretraining/configs/examples/debug/`:

- `gpt2_debug_cpu.yaml`: Small GPT-2 for testing
- `llama31_debug_cpu.yaml`: Small Llama 3.1 for testing
- `deepseek3_debug_cpu.yaml`: Small DeepSeek 3 for testing
