# MinLLM Pretraining

This directory contains the pretraining infrastructure for MinLLM, supporting GPT-2, Llama 3, and DeepSeek 3 architectures.

## Supported Models

| Model | Status |
|-------|--------|
| GPT-2 | ✅ |
| LLaMA 3.1 | ✅ |
| DeepSeek 3 | ✅ |
| Qwen 3 | ❌ |
| Kimi k2 | ❌ |

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

### Multi-GPU Training

```bash
# Single GPU (default)
uv run python pretraining/scripts/train_gpt2.py configs/examples/debug/gpt2_debug.yaml
uv run python pretraining/scripts/train_llama3.py configs/examples/debug/llama31_debug.yaml
uv run python pretraining/scripts/train_deepseek3.py configs/examples/debug/deepseek3_debug.yaml

# Multi-GPU with torchrun
uv run torchrun --nproc_per_node=4 pretraining/scripts/train_gpt2.py configs/examples/debug/gpt2_debug.yaml
uv run torchrun --nproc_per_node=4 pretraining/scripts/train_llama3.py configs/examples/debug/llama31_debug.yaml
uv run torchrun --nproc_per_node=4 pretraining/scripts/train_deepseek3.py configs/examples/debug/deepseek3_debug.yaml
```

For multi-GPU training, you must use torchrun and update the execution strategy in your config file to `ddp` instead of `single`.

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

All models are configured through YAML files. Example configurations are in `configs/examples/debug/`:

- `gpt2_debug.yaml`: Small GPT-2 for testing
- `llama31_debug.yaml`: Small Llama 3.1 for testing
- `deepseek3_debug.yaml`: Small DeepSeek 3 for testing

## Development

### Adding a New Architecture

1. Create architecture config in `configs/model/architectures/`
2. Implement model class in `common/patterns/architectures/`
3. Create transformer block in `common/patterns/blocks/`
4. Add training script in `scripts/`
5. Write tests in `tests/`

### Running a Custom Training

1. Create your YAML config file
2. Modify the training script to point to your config
3. Adjust hyperparameters as needed
4. Run the training script
