# MinLLM Test Suite

This directory contains comprehensive tests for the MinLLM pretraining framework, covering all major components from low-level patterns to high-level configurations.

## Test Structure

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── configs/           # Configuration parsing tests
│   │   └── test_config_parsing.py
│   └── patterns/          # Core pattern implementations
│       ├── components/    # Component tests (cache, position embeddings)
│       │   ├── test_cache.py
│       │   └── test_position.py
│       ├── test_attention.py         # Attention mechanisms (MHA, GQA, MLA)
│       ├── test_ffn.py               # Feed-forward networks (MLP, SwiGLU)
│       ├── test_moe.py               # Mixture of Experts
│       └── test_transformer_blocks.py # Architecture-specific transformer blocks
└── integration/           # End-to-end integration tests
    ├── test_generation.py           # Text generation tests
    ├── test_llm_forward.py          # Forward pass tests
    └── test_yaml_to_model_debug.py  # YAML config to model tests
```

## Running Tests

### Prerequisites

Make sure you're in the project root:
```bash
cd /path/to/minllm
```

The project uses `uv` for dependency management. All test commands use `uv run` to ensure the correct environment.

### Run All Tests

```bash
# Run all tests with summary
uv run pytest pretraining/tests/

# Run all tests with verbose output
uv run pytest pretraining/tests/ -v

# Run all tests with detailed output per test
uv run pytest pretraining/tests/ -vv
```

### Run Specific Test Categories

#### Unit Tests Only
```bash
# All unit tests
uv run pytest pretraining/tests/unit/ -v

# Pattern tests only
uv run pytest pretraining/tests/unit/patterns/ -v

# Component tests only
uv run pytest pretraining/tests/unit/patterns/components/ -v

# Config tests only
uv run pytest pretraining/tests/unit/configs/ -v
```

#### Integration Tests Only
```bash
# All integration tests
uv run pytest pretraining/tests/integration/ -v
```

### Run Specific Test Files

```bash
# Test attention mechanisms
uv run pytest pretraining/tests/unit/patterns/test_attention.py -v

# Test transformer blocks
uv run pytest pretraining/tests/unit/patterns/test_transformer_blocks.py -v

# Test MoE implementations
uv run pytest pretraining/tests/unit/patterns/test_moe.py -v

# Test FFN implementations
uv run pytest pretraining/tests/unit/patterns/test_ffn.py -v

# Test generation capabilities
uv run pytest pretraining/tests/integration/test_generation.py -v
```
