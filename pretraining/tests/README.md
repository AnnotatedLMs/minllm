# Test Suite Organization

This directory contains comprehensive tests for the minllm pretraining codebase.

## Directory Structure

### `unit/` - Unit Tests
Tests individual components in isolation.

#### `unit/patterns/components/`
- `test_position.py` - Position encoding tests (RoPE, PartialRoPE, LearnedPositionEmbedding)
- `test_cache.py` - KV cache functionality tests
- `test_heads.py` - Multi-token prediction head tests

#### `unit/patterns/`
- `test_attention.py` - Attention mechanism tests (MultiHeadAttention, GroupedQueryAttention, MultiHeadLatentAttention)
- `test_ffn.py` - Feedforward network tests (MLP, MultiplicativeGatedFFN)
- `test_moe.py` - Mixture of Experts tests (StandardMoE, AuxLossFreeMoE)
- `test_transformer.py` - Transformer block tests (BiasedLNTransformerBlock, RMSNormTransformerBlock)

#### `unit/base/`
- `test_initialization.py` - Weight initialization tests

### `integration/` - Integration Tests
Tests interactions between components and full workflows.

- `test_llm_forward.py` - Tests training_forward, inference_forward methods
- `test_generation.py` - Tests text generation functionality
- `test_kv_cache_flow.py` - Tests KV cache during generation
- `test_yaml_to_model.py` - Tests config to model creation

### `fixtures/` - Shared Test Fixtures
Reusable test configurations and utilities.

- `configs.py` - Test configuration objects
- `models.py` - Small model factories for testing

## Test Categories

### 1. Component Tests (Unit)
- **Position Encodings**: Test RoPE rotations, position offsets, scaling
- **Attention**: Test attention scores, masking, key-value handling
- **FFN**: Test gating mechanisms, activations
- **MoE**: Test routing, load balancing, auxiliary losses
- **Cache**: Test KV storage, updates, memory management

### 2. Model Tests (Integration)
- **Forward Passes**: Test training vs inference paths
- **Generation**: Test autoregressive generation with/without cache
- **Configuration**: Test YAML parsing and model instantiation
- **End-to-end**: Test full training step including loss computation

### 3. Performance Tests (Not Yet Implemented)
- Memory usage during generation
- Speed comparison with/without KV cache
- Scaling tests for different model sizes

## Running Tests

```bash
# Run all tests
pytest pretraining/tests/

# Run specific test file
pytest pretraining/tests/unit/patterns/components/test_position.py

# Run with coverage
pytest --cov=pretraining.common.patterns pretraining/tests/

# Run integration tests only
pytest pretraining/tests/integration/

# Run with markers (when implemented)
pytest -m "not slow" pretraining/tests/
```

## Writing New Tests

### Guidelines
1. Use descriptive test names that explain what is being tested
2. Include docstrings explaining the test purpose
3. Use fixtures for reusable components
4. Test both normal and edge cases
5. Use `torch.testing.assert_close` for tensor comparisons
6. Mock external dependencies when appropriate

### Example Test Structure
```python
class TestComponentName:
    """Test ComponentName functionality."""

    @pytest.fixture
    def component(self) -> ComponentType:
        """Create component instance for testing."""
        return ComponentType(...)

    def test_normal_case(self, component: ComponentType) -> None:
        """Test normal operation."""
        # Arrange
        input_data = ...

        # Act
        output = component(input_data)

        # Assert
        assert output.shape == expected_shape
        torch.testing.assert_close(output, expected_output)

    def test_edge_case(self, component: ComponentType) -> None:
        """Test edge case behavior."""
        with pytest.raises(ValueError):
            component(invalid_input)
```

## Test Coverage Goals

### Priority 1 (Critical)
- [ ] All forward methods (training_forward, inference_forward)
- [ ] Generation with and without KV cache
- [ ] Position encoding with offsets
- [ ] Attention masking
- [ ] Loss computation

### Priority 2 (Important)
- [ ] MoE routing and load balancing
- [ ] Multi-token prediction
- [ ] RoPE scaling
- [ ] Gradient flow tests
- [ ] Initialization tests

### Priority 3 (Nice to Have)
- [ ] Performance benchmarks
- [ ] Memory profiling
- [ ] Numerical stability tests
- [ ] Distributed training tests
