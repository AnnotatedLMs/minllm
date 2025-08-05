"""
Unit tests for position encoding components.

Tests RoPE, PartialRoPE, and LearnedPositionEmbedding implementations.
"""

# Third Party
import pytest
import torch
import torch.testing

# Project
from pretraining.common.patterns.position import core
from pretraining.common.patterns.position import learned
from pretraining.common.patterns.position import rope_partial
from pretraining.configs.model.components import position


class TestRoPE:
    """Test RoPE implementation."""

    @pytest.fixture
    def rope_module(self) -> core.PrecomputedRoPE:
        """Create a RoPE module."""
        return core.PrecomputedRoPE(
            dim=64,  # head_dim
            theta=10000.0,
            max_seq_len=128,
        )

    def test_rope_initialization(self, rope_module: core.PrecomputedRoPE) -> None:
        """Test RoPE initializes correctly."""
        assert rope_module.dim == 64
        assert rope_module.max_seq_len == 128

        # Check precomputed frequencies shape
        assert rope_module.freqs_cis.shape == (128, 32, 2)  # (max_seq_len, dim//2, 2)

        # Check inv_freq is registered as buffer
        assert hasattr(rope_module, "inv_freq")
        assert isinstance(rope_module.inv_freq, torch.Tensor)

    def test_rope_forward_basic(self, rope_module: core.PrecomputedRoPE) -> None:
        """Test basic forward pass."""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 8, 64
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Forward pass
        output = rope_module(x, position_offset=0)

        # Check output shape matches input
        assert output.shape == x.shape

        # Check output is different from input (rotation applied)
        assert not torch.allclose(output, x)

    def test_rope_position_offset(self, rope_module: core.PrecomputedRoPE) -> None:
        """Test position offset functionality for KV caching."""
        batch_size, num_heads, head_dim = 2, 8, 64

        # First token at position 0
        x1 = torch.randn(batch_size, 1, num_heads, head_dim)
        out1 = rope_module(x1, position_offset=0)

        # Second token at position 1
        x2 = torch.randn(batch_size, 1, num_heads, head_dim)
        out2 = rope_module(x2, position_offset=1)

        # Full sequence
        x_full = torch.cat([x1, x2], dim=1)
        out_full = rope_module(x_full, position_offset=0)

        # Outputs should match when processed separately vs together
        torch.testing.assert_close(out1, out_full[:, :1])
        torch.testing.assert_close(out2, out_full[:, 1:2])

    def test_rope_max_position_exceeded(self, rope_module: core.PrecomputedRoPE) -> None:
        """Test error when position exceeds precomputed max."""
        x = torch.randn(2, 10, 8, 64)

        # This should exceed max_seq_len of 128
        with pytest.raises(ValueError, match="exceeds precomputed max"):
            rope_module(x, position_offset=120)  # 120 + 10 > 128

    def test_rope_deterministic(self, rope_module: core.PrecomputedRoPE) -> None:
        """Test RoPE is deterministic."""
        x = torch.randn(2, 10, 8, 64)

        out1 = rope_module(x, position_offset=0)
        out2 = rope_module(x, position_offset=0)

        torch.testing.assert_close(out1, out2)


class TestPartialRoPE:
    """Test PartialRoPE implementation for DeepSeek."""

    @pytest.fixture
    def partial_rope(self) -> rope_partial.PartialRoPE:
        """Create a PartialRoPE module."""
        return rope_partial.PartialRoPE(
            dim=64,  # rope_dim, not full head_dim
            theta=10000.0,
        )

    def test_partial_rope_forward(self, partial_rope: rope_partial.PartialRoPE) -> None:
        """Test PartialRoPE forward pass."""
        batch_size, num_heads, seq_len, rope_dim = 2, 8, 10, 64
        x = torch.randn(batch_size, num_heads, seq_len, rope_dim)

        output = partial_rope(x)

        # Check output shape
        assert output.shape == x.shape

        # Check rotation applied
        assert not torch.allclose(output, x)

    def test_partial_rope_no_position_offset(self, partial_rope: rope_partial.PartialRoPE) -> None:
        """Test PartialRoPE doesn't accept position_offset."""
        x = torch.randn(2, 8, 10, 64)

        # Should work without position_offset
        output = partial_rope(x)
        assert output.shape == x.shape

        # Note: PartialRoPE.forward() doesn't accept position_offset parameter
        # This is by design - DeepSeek computes positions dynamically


class TestLearnedPositionEmbedding:
    """Test learned position embeddings."""

    @pytest.fixture
    def learned_pos_emb(self) -> learned.LearnedPositionEmbedding:
        """Create learned position embedding module."""
        return learned.LearnedPositionEmbedding(
            max_position_embeddings=1024,
            embedding_dim=768,
            init_std=0.02,
        )

    def test_learned_embedding_forward(
        self, learned_pos_emb: learned.LearnedPositionEmbedding
    ) -> None:
        """Test learned embedding forward pass."""
        batch_size, seq_len = 2, 100
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        embeddings = learned_pos_emb(position_ids)

        # Check shape
        assert embeddings.shape == (batch_size, seq_len, 768)

        # Check embeddings are learnable parameters
        assert learned_pos_emb.wpe.weight.requires_grad

    def test_learned_embedding_out_of_range(
        self, learned_pos_emb: learned.LearnedPositionEmbedding
    ) -> None:
        """Test error when position exceeds max."""
        position_ids = torch.tensor([[0, 500, 1024]])  # 1024 >= max_position_embeddings

        with pytest.raises(ValueError, match="Position ids must be less than"):
            learned_pos_emb(position_ids)


class TestRoPEScaling:
    """Test RoPE scaling for extended context."""

    def test_rope_with_scaling(self) -> None:
        """Test RoPE with scaling configuration."""
        scaling_config = position.RoPEScalingConfig(
            scale_factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_context_len=8192,
        )

        rope_module = core.PrecomputedRoPE(
            dim=128,
            theta=500000.0,
            max_seq_len=65536,  # Extended context
            scaling=scaling_config,
        )

        # Test forward pass with extended sequence
        x = torch.randn(1, 1000, 32, 128)
        output = rope_module(x, position_offset=0)

        assert output.shape == x.shape
