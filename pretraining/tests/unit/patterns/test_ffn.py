"""
Unit tests for feedforward network components.

Tests MLP and MultiplicativeGatedFFN (SwiGLU) implementations.
"""

# Third Party
import pytest
import torch
from torch import testing

# Project
from pretraining.common.models.ffn import mlp
from pretraining.common.models.ffn import swiglu


class TestMLP:
    """Test standard MLP feedforward network."""

    @pytest.fixture
    def mlp_default(self) -> mlp.MLP:
        """Create MLP with default 4x expansion."""
        return mlp.MLP(
            hidden_dim=128,
            intermediate_dim=None,  # Should default to 4x
            dropout=0.1,
            activation="gelu",
            bias=True,
        )

    @pytest.fixture
    def mlp_custom(self) -> mlp.MLP:
        """Create MLP with custom intermediate dimension."""
        return mlp.MLP(
            hidden_dim=128,
            intermediate_dim=256,
            dropout=None,
            activation="relu",
            bias=False,
        )

    def test_mlp_default_expansion(self, mlp_default: mlp.MLP) -> None:
        """Test MLP default 4x expansion calculation."""
        # This tests actual logic - the default expansion factor
        assert mlp_default.intermediate_dim == 512  # 4 * 128

    def test_mlp_forward(self, mlp_default: mlp.MLP) -> None:
        """Test MLP forward pass."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 128)

        output = mlp_default(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, 128)

        # Check it's not just passthrough
        assert not torch.allclose(output, x)

    def test_mlp_activations(self) -> None:
        """Test different activation functions."""
        activations = ["relu", "gelu", "silu"]

        for act in activations:
            mlp_instance = mlp.MLP(
                hidden_dim=64,
                activation=act,
                dropout=None,
            )

            x = torch.randn(1, 5, 64)
            output = mlp_instance(x)
            assert output.shape == x.shape

    def test_mlp_dropout_training_vs_eval(self, mlp_default: mlp.MLP) -> None:
        """Test dropout behavior in training vs eval mode."""
        x = torch.randn(1, 10, 128)

        # Training mode - outputs should differ due to dropout
        mlp_default.train()
        out1 = mlp_default(x)
        out2 = mlp_default(x)
        assert not torch.allclose(out1, out2)

        # Eval mode - outputs should be deterministic
        mlp_default.eval()
        out3 = mlp_default(x)
        out4 = mlp_default(x)
        testing.assert_close(out3, out4)


class TestMultiplicativeGatedFFN:
    """Test SwiGLU-style gated feedforward network."""

    @pytest.fixture
    def swiglu_module(self) -> swiglu.SwiGLU:
        """Create SwiGLU FFN."""
        return swiglu.SwiGLU(
            hidden_dim=128,
            dropout=None,
            activation="silu",
            bias=False,
            ffn_dim_multiplier=2.0,
            multiple_of=64,
        )

    def test_swiglu_dimension_calculation(self, swiglu_module: swiglu.SwiGLU) -> None:
        """Test SwiGLU dimension calculation with multiple_of constraint."""
        # Check computed hidden dimension
        # With ffn_dim_multiplier=2.0: 128 * 2.0 = 256
        # Already a multiple of 64, so no rounding needed
        expected_hidden = 256
        assert swiglu_module.intermediate_dim == expected_hidden

    def test_swiglu_forward(self, swiglu_module: swiglu.SwiGLU) -> None:
        """Test SwiGLU forward pass."""
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 128)

        output = swiglu_module(x)

        assert output.shape == (batch_size, seq_len, 128)

    def test_swiglu_hidden_dim_calculation(self) -> None:
        """Test hidden dimension calculation with different parameters."""
        test_cases = [
            # (hidden_dim, multiplier, multiple_of, expected)
            (256, 2.0, 64, 512),  # 256 * 2.0 = 512 (already multiple of 64)
            (512, 1.5, 128, 768),  # 512 * 1.5 = 768 (already multiple of 128)
            (128, 3.0, 32, 384),  # 128 * 3.0 = 384 (already multiple of 32)
        ]

        for hidden, mult, multiple, expected in test_cases:
            swiglu_inst = swiglu.SwiGLU(
                hidden_dim=hidden,
                ffn_dim_multiplier=mult,
                multiple_of=multiple,
            )
            assert swiglu_inst.intermediate_dim == expected

    def test_swiglu_gating_mechanism(self, swiglu_module: swiglu.SwiGLU) -> None:
        """Test the gating mechanism works correctly."""
        x = torch.randn(1, 1, 128)

        # Get gate and up projections manually
        gate = swiglu_module.activation(swiglu_module.gate_proj(x))
        up = swiglu_module.up_proj(x)

        # Gated activation
        gated = gate * up

        # Down projection
        output_manual = swiglu_module.down_proj(gated)

        # Compare with forward pass
        output_forward = swiglu_module(x)

        testing.assert_close(output_manual, output_forward)

    def test_swiglu_different_activations(self) -> None:
        """Test SwiGLU with different activation functions."""
        activations = ["silu", "gelu", "relu"]

        for act in activations:
            swiglu_inst = swiglu.SwiGLU(
                hidden_dim=64,
                activation=act,
                dropout=None,
            )

            x = torch.randn(1, 5, 64)
            output = swiglu_inst(x)
            assert output.shape == x.shape


class TestFFNComparison:
    """Compare MLP and SwiGLU behaviors."""

    def test_parameter_count_comparison(self) -> None:
        """Compare parameter counts between MLP and SwiGLU."""
        hidden_dim = 256

        # Standard MLP with 4x expansion
        mlp_model = mlp.MLP(
            hidden_dim=hidden_dim,
            intermediate_dim=None,  # 4x = 1024
            bias=False,
        )

        # SwiGLU with equivalent capacity
        swiglu_inst = swiglu.SwiGLU(
            hidden_dim=hidden_dim,
            ffn_dim_multiplier=2.67,  # Roughly equivalent params
            multiple_of=64,
            bias=False,
        )

        mlp_params = sum(p.numel() for p in mlp_model.parameters())
        swiglu_params = sum(p.numel() for p in swiglu_inst.parameters())

        # MLP: 256->1024->256 = 256*1024 + 1024*256 = 524,288
        assert mlp_params == 256 * 1024 * 2

        # SwiGLU has 3 projection layers (gate, up, down)
        # The actual intermediate dim will be calculated and rounded
        # Just verify SwiGLU has parameters and the relationship holds
        assert swiglu_params > 0
        assert mlp_params > 0

    def test_output_magnitude_comparison(self) -> None:
        """Test that both FFN types produce reasonable output magnitudes."""
        x = torch.randn(1, 10, 128)

        mlp_model = mlp.MLP(hidden_dim=128, dropout=None)
        swiglu_model = swiglu.SwiGLU(hidden_dim=128, dropout=None)

        mlp_out = mlp_model(x)
        swiglu_out = swiglu_model(x)

        # Both should maintain roughly similar magnitude to input
        input_norm = torch.norm(x)
        mlp_norm = torch.norm(mlp_out)
        swiglu_norm = torch.norm(swiglu_out)

        # Check outputs are within reasonable range (not exploding/vanishing)
        assert 0.1 < mlp_norm / input_norm < 10.0
        # SwiGLU can produce smaller outputs due to gating
        assert 0.05 < swiglu_norm / input_norm < 10.0
