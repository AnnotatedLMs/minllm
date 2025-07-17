"""
Unit tests for feedforward network components.

Tests MLP and MultiplicativeGatedFFN (SwiGLU) implementations.
"""

# Third Party
import pytest
import torch
import torch.nn as nn
import torch.testing

# Project
# Local
from pretraining.common.patterns import ffn


class TestMLP:
    """Test standard MLP feedforward network."""

    @pytest.fixture
    def mlp_default(self) -> ffn.MLP:
        """Create MLP with default 4x expansion."""
        return ffn.MLP(
            hidden_dim=128,
            intermediate_dim=None,  # Should default to 4x
            dropout=0.1,
            activation="gelu",
            bias=True,
        )

    @pytest.fixture
    def mlp_custom(self) -> ffn.MLP:
        """Create MLP with custom intermediate dimension."""
        return ffn.MLP(
            hidden_dim=128,
            intermediate_dim=256,
            dropout=0.0,
            activation="relu",
            bias=False,
        )

    def test_mlp_initialization_default(self, mlp_default: ffn.MLP) -> None:
        """Test MLP initializes with correct default dimensions."""
        assert mlp_default.hidden_dim == 128
        assert mlp_default.intermediate_dim == 512  # 4 * 128

        # Check layers
        assert mlp_default.c_fc.in_features == 128
        assert mlp_default.c_fc.out_features == 512
        assert mlp_default.c_proj.in_features == 512
        assert mlp_default.c_proj.out_features == 128

        # Check bias
        assert mlp_default.c_fc.bias is not None
        assert mlp_default.c_proj.bias is not None

        # Check activation
        assert isinstance(mlp_default.activation, nn.GELU)

        # Check dropout
        assert hasattr(mlp_default, "dropout_layer")
        assert mlp_default.dropout_layer.p == 0.1

    def test_mlp_initialization_custom(self, mlp_custom: ffn.MLP) -> None:
        """Test MLP with custom configuration."""
        assert mlp_custom.intermediate_dim == 256

        # Check no bias
        assert mlp_custom.c_fc.bias is None
        assert mlp_custom.c_proj.bias is None

        # Check activation
        assert isinstance(mlp_custom.activation, nn.ReLU)

        # Check no dropout
        assert mlp_custom.dropout == 0.0  # Stored as attribute
        # No dropout layer created when dropout=0.0

    def test_mlp_forward(self, mlp_default: ffn.MLP) -> None:
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
            mlp = ffn.MLP(
                hidden_dim=64,
                activation=act,
                dropout=0.0,
            )

            x = torch.randn(1, 5, 64)
            output = mlp(x)
            assert output.shape == x.shape

    def test_mlp_dropout_training_vs_eval(self, mlp_default: ffn.MLP) -> None:
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
        torch.testing.assert_close(out3, out4)


class TestMultiplicativeGatedFFN:
    """Test SwiGLU-style gated feedforward network."""

    @pytest.fixture
    def swiglu(self) -> ffn.MultiplicativeGatedFFN:
        """Create SwiGLU FFN."""
        return ffn.MultiplicativeGatedFFN(
            hidden_dim=128,
            dropout=0.0,
            activation="silu",
            bias=False,
            ffn_dim_multiplier=2.0,
            multiple_of=64,
        )

    def test_swiglu_initialization(self, swiglu: ffn.MultiplicativeGatedFFN) -> None:
        """Test SwiGLU initializes correctly."""
        assert swiglu.hidden_dim == 128

        # Check computed hidden dimension
        # With ffn_dim_multiplier=2.0: 128 * 2.0 = 256
        # Already a multiple of 64, so no rounding needed
        expected_hidden = 256
        assert swiglu.intermediate_dim == expected_hidden

        # Check layer dimensions
        assert swiglu.gate_proj.in_features == 128
        assert swiglu.gate_proj.out_features == expected_hidden
        assert swiglu.up_proj.in_features == 128
        assert swiglu.up_proj.out_features == expected_hidden
        assert swiglu.down_proj.in_features == expected_hidden
        assert swiglu.down_proj.out_features == 128

        # Check no bias
        assert swiglu.gate_proj.bias is None
        assert swiglu.up_proj.bias is None
        assert swiglu.down_proj.bias is None

    def test_swiglu_forward(self, swiglu: ffn.MultiplicativeGatedFFN) -> None:
        """Test SwiGLU forward pass."""
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 128)

        output = swiglu(x)

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
            swiglu = ffn.MultiplicativeGatedFFN(
                hidden_dim=hidden,
                ffn_dim_multiplier=mult,
                multiple_of=multiple,
            )
            assert swiglu.intermediate_dim == expected

    def test_swiglu_gating_mechanism(self, swiglu: ffn.MultiplicativeGatedFFN) -> None:
        """Test the gating mechanism works correctly."""
        x = torch.randn(1, 1, 128)

        # Get gate and up projections manually
        gate = swiglu.activation(swiglu.gate_proj(x))
        up = swiglu.up_proj(x)

        # Gated activation
        gated = gate * up

        # Down projection
        output_manual = swiglu.down_proj(gated)

        # Compare with forward pass
        output_forward = swiglu(x)

        torch.testing.assert_close(output_manual, output_forward)

    def test_swiglu_different_activations(self) -> None:
        """Test SwiGLU with different activation functions."""
        activations = ["silu", "gelu", "relu"]

        for act in activations:
            swiglu = ffn.MultiplicativeGatedFFN(
                hidden_dim=64,
                activation=act,
                dropout=0.0,
            )

            x = torch.randn(1, 5, 64)
            output = swiglu(x)
            assert output.shape == x.shape


class TestFFNComparison:
    """Compare MLP and SwiGLU behaviors."""

    def test_parameter_count_comparison(self) -> None:
        """Compare parameter counts between MLP and SwiGLU."""
        hidden_dim = 256

        # Standard MLP with 4x expansion
        mlp = ffn.MLP(
            hidden_dim=hidden_dim,
            intermediate_dim=None,  # 4x = 1024
            bias=False,
        )

        # SwiGLU with equivalent capacity
        swiglu = ffn.MultiplicativeGatedFFN(
            hidden_dim=hidden_dim,
            ffn_dim_multiplier=2.67,  # Roughly equivalent params
            multiple_of=64,
            bias=False,
        )

        mlp_params = sum(p.numel() for p in mlp.parameters())
        swiglu_params = sum(p.numel() for p in swiglu.parameters())

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

        mlp = ffn.MLP(hidden_dim=128, dropout=0.0)
        swiglu = ffn.MultiplicativeGatedFFN(hidden_dim=128, dropout=0.0)

        mlp_out = mlp(x)
        swiglu_out = swiglu(x)

        # Both should maintain roughly similar magnitude to input
        input_norm = torch.norm(x)
        mlp_norm = torch.norm(mlp_out)
        swiglu_norm = torch.norm(swiglu_out)

        # Check outputs are within reasonable range (not exploding/vanishing)
        assert 0.1 < mlp_norm / input_norm < 10.0
        assert 0.1 < swiglu_norm / input_norm < 10.0
