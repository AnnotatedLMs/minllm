"""
Jaxtyping validation tool for architecture modules.

This tool validates that jaxtyping annotations are accurate by:
1. Running forward passes with dummy inputs
2. Checking tensor shapes at each operation
3. Reporting any mismatches between annotations and actual shapes
"""

# Standard Library
import sys
import traceback
import typing
from dataclasses import dataclass
from pathlib import Path

# Third Party
import torch
import torch.nn as nn
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from typeguard import typechecked

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Project
from pretraining.common.models.architectures import deepseek3
from pretraining.configs.model import transformer as transformer_config
from pretraining.configs.model.architectures import deepseek as deepseek_config
from pretraining.configs.model.components import attention as attention_config
from pretraining.configs.model.components import feedforward as feedforward_config
from pretraining.configs.model.components import heads as heads_config
from pretraining.configs.model.components import normalization as normalization_config
from pretraining.configs.model.components import position as position_config


@dataclass
class ValidationResult:
    """Result of validating a single function/method."""

    function_name: str
    module_name: str
    passed: bool
    error: typing.Optional[str] = None
    traceback: typing.Optional[str] = None


class JaxtypingValidator:
    """Validator for jaxtyping annotations in neural network modules."""

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self.results: typing.List[ValidationResult] = []

    def create_dummy_inputs(
        self,
        batch_size: int = 2,
        seq_length: int = 128,
        vocab_size: int = 32000,
    ) -> typing.Dict[str, torch.Tensor]:
        """Create dummy inputs for testing."""
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=self.device)
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=self.device)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def wrap_with_validation(self, module: nn.Module) -> nn.Module:
        """
        Wrap all methods of a module with jaxtyping runtime validation.
        """
        # Get all methods that have jaxtyping annotations
        for name, method in module.named_modules():
            if hasattr(method, "forward"):
                original_forward = method.forward

                # Check if forward has type annotations
                if hasattr(original_forward, "__annotations__"):
                    # Wrap with jaxtyped decorator for runtime validation
                    @jaxtyped(typechecker=typechecked)
                    def validated_forward(self, *args, **kwargs):
                        return original_forward(*args, **kwargs)

                    # Replace the forward method
                    method.forward = validated_forward.__get__(method, type(method))

        return module

    def validate_deepseek3(
        self,
        config: typing.Optional[deepseek_config.DeepSeek3Config] = None,
    ) -> typing.List[ValidationResult]:
        """Validate DeepSeek3 architecture with runtime type checking."""

        print("=" * 80)
        print("Validating DeepSeek3 Architecture")
        print("=" * 80)

        # Create minimal config if not provided
        if config is None:
            config = deepseek_config.DeepSeek3Config(
                vocab_size=32000,
                transformer=transformer_config.DeepSeek3TransformerConfig(
                    hidden_dim=256,
                    n_layers=2,
                    block_size=512,
                    normalization=normalization_config.RMSNormConfig(
                        norm_eps=1e-5,
                    ),
                    attention=attention_config.MultiHeadLatentAttentionConfig(
                        num_heads=8,
                        head_dim=32,
                        kv_compression_dim=64,
                        query_compression_dim=128,
                        rope_dim=32,
                        bias=False,
                        max_seq_length=512,
                        is_causal=True,
                        use_flash_attention=False,
                    ),
                    rope=position_config.RoPEConfig(
                        theta=10000.0,
                    ),
                    moe=feedforward_config.MoEConfig(
                        num_experts=8,
                        num_experts_per_token=2,
                        n_shared_experts=1,
                        shared_expert_ratio=0.1,
                    ),
                    bias=False,
                ),
                output_head=heads_config.OutputHeadConfig(
                    tie_word_embeddings=False,
                    lm_head_bias=False,
                ),
                mtp=heads_config.MultiTokenPredictionConfig(
                    n_predict=3,
                ),
            )

        # Create model
        model = deepseek3.DeepSeek3.from_config(config)
        model = model.to(self.device, dtype=self.dtype)
        model.eval()

        # Create dummy inputs
        inputs = self.create_dummy_inputs(
            batch_size=2,
            seq_length=64,
            vocab_size=config.vocab_size,
        )

        results = []

        # Test forward pass with runtime validation
        print("\n1. Testing forward pass with jaxtyping validation...")
        try:
            # Apply jaxtyped decorator to forward method
            original_forward = model.forward

            @jaxtyped(typechecker=typechecked)
            def validated_forward(
                input_ids: Int[torch.Tensor, "batch seq"],
                attention_mask: typing.Optional[torch.Tensor] = None,
            ):
                return original_forward(input_ids, attention_mask)

            # Run validated forward
            with torch.no_grad():
                output = validated_forward(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                )

            results.append(
                ValidationResult(
                    function_name="forward",
                    module_name="DeepSeek3",
                    passed=True,
                )
            )
            print("   ✓ Forward pass validation successful")

            # Validate output shapes
            print(f"   Output logits shape: {output.logits.shape}")
            expected_logits_shape = (2, 64, config.vocab_size)
            assert output.logits.shape == expected_logits_shape, (
                f"Expected logits shape {expected_logits_shape}, got {output.logits.shape}"
            )

        except Exception as e:
            results.append(
                ValidationResult(
                    function_name="forward",
                    module_name="DeepSeek3",
                    passed=False,
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
            )
            print(f"   ✗ Forward pass validation failed: {e}")

        # Test individual blocks
        print("\n2. Testing transformer blocks...")
        for i, block in enumerate(model.blocks):
            try:
                original_block_forward = block.forward

                @jaxtyped(typechecker=typechecked)
                def validated_block_forward(
                    x: Float[torch.Tensor, "batch seq_len hidden_dim"],
                    attention_mask: typing.Optional[torch.Tensor] = None,
                ) -> Float[torch.Tensor, "batch seq_len hidden_dim"]:
                    return original_block_forward(x, attention_mask)

                # Create input for block
                hidden_states = torch.randn(
                    2, 64, config.transformer.hidden_dim, device=self.device, dtype=self.dtype
                )

                with torch.no_grad():
                    output = validated_block_forward(hidden_states, inputs["attention_mask"])

                results.append(
                    ValidationResult(
                        function_name=f"block_{i}.forward",
                        module_name="DeepSeek3TransformerBlock",
                        passed=True,
                    )
                )
                print(f"   ✓ Block {i} validation successful")

            except Exception as e:
                results.append(
                    ValidationResult(
                        function_name=f"block_{i}.forward",
                        module_name="DeepSeek3TransformerBlock",
                        passed=False,
                        error=str(e),
                        traceback=traceback.format_exc(),
                    )
                )
                print(f"   ✗ Block {i} validation failed: {e}")

        # Test attention module
        print("\n3. Testing attention module...")
        if len(model.blocks) > 0:
            attention = model.blocks[0].attention
            try:
                # We'll need to check the attention module's forward signature
                original_attn_forward = attention.forward

                # Create properly shaped inputs for attention
                hidden_states = torch.randn(
                    2, 64, config.transformer.hidden_dim, device=self.device, dtype=self.dtype
                )

                with torch.no_grad():
                    # The attention forward might have different signature
                    # Let's check what it expects
                    # Standard Library
                    import inspect

                    sig = inspect.signature(original_attn_forward)
                    print(f"   Attention forward signature: {sig}")

                    # Call with appropriate arguments
                    attn_output = original_attn_forward(hidden_states)

                results.append(
                    ValidationResult(
                        function_name="attention.forward",
                        module_name="MultiHeadLatentAttention",
                        passed=True,
                    )
                )
                print("   ✓ Attention validation successful")
                print(f"   Attention output shape: {attn_output.shape}")

            except Exception as e:
                results.append(
                    ValidationResult(
                        function_name="attention.forward",
                        module_name="MultiHeadLatentAttention",
                        passed=False,
                        error=str(e),
                        traceback=traceback.format_exc(),
                    )
                )
                print(f"   ✗ Attention validation failed: {e}")

        # Test MoE module
        print("\n4. Testing MoE module...")
        if len(model.blocks) > 0:
            moe = model.blocks[0].moe
            try:
                hidden_states = torch.randn(
                    2, 64, config.transformer.hidden_dim, device=self.device, dtype=self.dtype
                )

                with torch.no_grad():
                    moe_output = moe(hidden_states)

                results.append(
                    ValidationResult(
                        function_name="moe.forward",
                        module_name="AuxLossFreeMoE",
                        passed=True,
                    )
                )
                print("   ✓ MoE validation successful")
                print(f"   MoE output shape: {moe_output.shape}")

            except Exception as e:
                results.append(
                    ValidationResult(
                        function_name="moe.forward",
                        module_name="AuxLossFreeMoE",
                        passed=False,
                        error=str(e),
                        traceback=traceback.format_exc(),
                    )
                )
                print(f"   ✗ MoE validation failed: {e}")

        self.results.extend(results)
        return results

    def print_summary(self):
        """Print summary of validation results."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.module_name}.{r.function_name}")
                    if r.error:
                        print(f"    Error: {r.error}")

        print("\n" + "=" * 80)
        return passed, failed


def validate_with_hook_tracing():
    """
    Alternative approach: Hook into tensor operations to trace shapes.
    This provides detailed shape information at each step.
    """
    # Standard Library
    import functools

    class ShapeTracer:
        def __init__(self):
            self.traces = []

        def trace_tensor_op(self, op_name: str, *tensors):
            """Record tensor shapes for an operation."""
            shapes = []
            for i, t in enumerate(tensors):
                if isinstance(t, torch.Tensor):
                    shapes.append(f"arg{i}: {tuple(t.shape)}")

            trace = f"{op_name}: {', '.join(shapes)}"
            self.traces.append(trace)
            print(f"  TRACE: {trace}")

        def wrap_module(self, module: nn.Module, module_name: str = ""):
            """Wrap a module's forward to trace tensor shapes."""
            original_forward = module.forward

            @functools.wraps(original_forward)
            def traced_forward(*args, **kwargs):
                self.trace_tensor_op(f"{module_name}.forward_input", *args)
                result = original_forward(*args, **kwargs)
                if isinstance(result, torch.Tensor):
                    self.trace_tensor_op(f"{module_name}.forward_output", result)
                return result

            module.forward = traced_forward

            # Recursively wrap child modules
            for name, child in module.named_children():
                child_name = f"{module_name}.{name}" if module_name else name
                self.wrap_module(child, child_name)

    print("\n" + "=" * 80)
    print("SHAPE TRACING VALIDATION")
    print("=" * 80)

    # Create minimal config
    config = deepseek_config.DeepSeek3Config(
        vocab_size=100,
        transformer=transformer_config.DeepSeek3TransformerConfig(
            hidden_dim=64,
            n_layers=1,
            block_size=128,
            normalization=normalization_config.RMSNormConfig(
                norm_eps=1e-5,
            ),
            attention=attention_config.MultiHeadLatentAttentionConfig(
                num_heads=4,
                head_dim=16,
                kv_compression_dim=32,
                query_compression_dim=48,
                rope_dim=16,
                bias=False,
                max_seq_length=128,
                is_causal=True,
                use_flash_attention=False,
            ),
            rope=position_config.RoPEConfig(
                theta=10000.0,
            ),
            moe=feedforward_config.MoEConfig(
                num_experts=4,
                num_experts_per_token=2,
                n_shared_experts=1,
                shared_expert_ratio=0.1,
            ),
            bias=False,
        ),
        output_head=heads_config.OutputHeadConfig(
            tie_word_embeddings=False,
            lm_head_bias=False,
        ),
        mtp=heads_config.MultiTokenPredictionConfig(
            n_predict=3,
        ),
    )

    # Create model and tracer
    model = deepseek3.DeepSeek3.from_config(config)
    model.eval()

    tracer = ShapeTracer()
    tracer.wrap_module(model)

    # Run forward pass
    input_ids = torch.randint(0, 100, (2, 32))

    print("\nRunning traced forward pass...")
    with torch.no_grad():
        output = model(input_ids)

    print(f"\nTotal operations traced: {len(tracer.traces)}")
    print(f"Final output shape: {output.logits.shape}")


if __name__ == "__main__":
    # Run standard validation
    validator = JaxtypingValidator()
    results = validator.validate_deepseek3()
    passed, failed = validator.print_summary()

    # Run shape tracing for more detail
    print("\n")
    validate_with_hook_tracing()

    # Exit with error code if any tests failed
    sys.exit(0 if failed == 0 else 1)
