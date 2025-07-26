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

### Run Specific Test Classes or Methods

```bash
# Run all GroupedQueryAttention tests
uv run pytest pretraining/tests/unit/patterns/test_attention.py::TestGroupedQueryAttention -v

# Run a specific test method
uv run pytest pretraining/tests/unit/patterns/test_attention.py::TestGroupedQueryAttention::test_gqa_with_kv_cache -v

# Run tests matching a pattern
uv run pytest pretraining/tests/ -k "attention" -v
```

### Useful pytest Options

```bash
# Stop on first failure
uv run pytest pretraining/tests/ -x

# Stop on first failure with verbose output and print statements
uv run pytest pretraining/tests/ -xvs

# Run only failed tests from last run
uv run pytest pretraining/tests/ --lf

# Run failed tests first, then the rest
uv run pytest pretraining/tests/ --ff

# Show local variables on failure
uv run pytest pretraining/tests/ -l

# Shorter traceback format
uv run pytest pretraining/tests/ --tb=short

# Show top 10 slowest tests
uv run pytest pretraining/tests/ --durations=10

# Run tests in parallel (requires pytest-xdist)
# uv run pytest pretraining/tests/ -n auto
```

## Test Coverage Summary

### Unit Tests (73 tests)

#### Configuration Tests (11 tests)
- **Config Parsing** (5 tests): GPT-2, Llama, DeepSeek YAML parsing
- **Config Validation** (3 tests): Required fields, incompatible combinations
- **Weight Init** (3 tests): Initialization strategies and validation

#### Component Tests (16 tests)
- **KV Cache** (6 tests): Buffer allocation, token updates, generation, reset
- **Position Embeddings** (10 tests):
  - Learned embeddings: initialization, forward, out-of-range
  - RoPE: basic, position offsets, scaling
  - PartialRoPE: dimension handling, MLA integration

#### Pattern Tests (46 tests)
- **Attention** (18 tests):
  - MultiHeadAttention (5): initialization, QKV, reshaping, causal masking
  - GroupedQueryAttention (5): GQA ratios, RoPE, KV cache
  - MultiHeadLatentAttention (5): compression, partial RoPE
  - Utilities (3): Flash Attention, masking, scaling

- **Feed-Forward Networks** (10 tests):
  - MLP (5): initialization, activations, dropout
  - MultiplicativeGatedFFN (5): SwiGLU, gating, dimension calculation

- **Mixture of Experts** (10 tests):
  - AuxLossFreeMoE (8): shared expert, gating bias, load tracking
  - Helpers (2): capacity calculation, edge cases

- **Transformer Blocks** (8 tests):
  - GPT2TransformerBlock (2): LayerNorm + MHA + MLP
  - Llama3TransformerBlock (2): RMSNorm + GQA + SwiGLU
  - DeepSeek3TransformerBlock (2): RMSNorm + MLA + MoE
  - Consistency (2): residual connections, gradient flow

### Integration Tests (15 tests)

- **Generation** (3 tests): GPT-2 and Llama generation, reproducibility
- **Forward Methods** (8 tests): Training/inference forward, KV cache, attention masks
- **YAML to Model** (4 tests): Config loading and model creation for each architecture

### Total: 88 tests (87 pass, 1 skipped)

## Common Test Patterns

### Running a Quick Smoke Test
```bash
# Run a single fast test to verify setup
uv run pytest pretraining/tests/unit/patterns/test_attention.py::TestMultiHeadAttention::test_mha_initialization -v
```

### Debugging a Failing Test
```bash
# Run with print statements visible and stop on failure
uv run pytest path/to/test.py::TestClass::test_method -xvs

# Run with full traceback
uv run pytest path/to/test.py::TestClass::test_method -xvs --tb=long

# Run with pdb on failure
uv run pytest path/to/test.py::TestClass::test_method -xvs --pdb
```

### Testing After Code Changes
```bash
# Run tests for the module you changed
uv run pytest pretraining/tests/unit/patterns/test_attention.py -v

# Then run integration tests to ensure nothing broke
uv run pytest pretraining/tests/integration/ -v

# Finally run all tests before committing
uv run pytest pretraining/tests/ -v
```

## Writing New Tests

When adding new tests:

1. **Location**: Place tests in the appropriate subdirectory
2. **Naming**: Use `test_*.py` for test files, `Test*` for test classes
3. **Structure**: Group related tests in classes
4. **Fixtures**: Use pytest fixtures for reusable test components
5. **Assertions**: Use `torch.testing.assert_close()` for tensor comparisons
6. **Documentation**: Add docstrings explaining what each test validates

Example test structure:
```python
class TestNewComponent:
    """Test new component functionality."""

    @pytest.fixture
    def component(self):
        """Create component for testing."""
        return NewComponent(...)

    def test_initialization(self, component):
        """Test proper initialization."""
        assert component.param == expected_value

    def test_forward_pass(self, component):
        """Test forward pass computation."""
        output = component(input)
        torch.testing.assert_close(output, expected_output)
```

## Important Notes

### Platform Compatibility
- Tests run on CPU by default (no CUDA required)
- All tests should pass on macOS, Linux, and Windows
- Flash Attention tests adapt based on availability

### Test Philosophy
Tests focus on:
1. **Shape Correctness**: Ensuring dimensions flow correctly
2. **Module Initialization**: Components created with expected parameters
3. **Basic Functionality**: Forward passes work without crashes
4. **Edge Cases**: Single tokens, full caches, extreme scenarios
5. **Determinism**: Reproducible outputs with fixed seeds

Tests do NOT check:
1. Exact numerical values (too dependent on initialization)
2. Training convergence (that's for actual training runs)
3. Model quality (tests ensure code works, not that models are good)

### Configuration
The project uses `pyproject.toml` for pytest configuration:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["pretraining/tests"]
```

This ensures tests can import from `pretraining.*` without PYTHONPATH manipulation.
