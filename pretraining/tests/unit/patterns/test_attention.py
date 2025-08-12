"""
Unit tests for attention mechanisms.

Tests MultiHeadAttention, GroupedQueryAttention, and MultiHeadLatentAttention.
"""

# Third Party
import pytest
import torch
from torch import testing

# Project
from pretraining.common.patterns.attention import grouped_query
from pretraining.common.patterns.attention import multi_head
from pretraining.common.patterns.attention import multi_latent
from pretraining.common.patterns.cache import kv_cache
from pretraining.common.patterns.position import core
from pretraining.common.patterns.position import rope_partial


# Helper to check Flash Attention availability
def has_flash_attention() -> bool:
    """Check if Flash Attention is available on this system."""
    return hasattr(torch.nn.functional, "scaled_dot_product_attention")


# Pytest marker for Flash Attention tests
pytestmark_flash = pytest.mark.skipif(
    has_flash_attention(), reason="Test requires manual attention (Flash Attention is available)"
)

pytestmark_no_flash = pytest.mark.skipif(
    not has_flash_attention(), reason="Test requires Flash Attention"
)


class TestMultiHeadAttention:
    """Test standard multi-head attention (GPT-2 style)."""

    @pytest.fixture
    def mha(self) -> multi_head.MultiHeadAttention:
        """Create MultiHeadAttention instance."""
        return multi_head.MultiHeadAttention(
            hidden_dim=128,
            num_heads=4,
            dropout=None,
            bias=True,
            max_seq_length=512,
            is_causal=True,
        )

    def test_mha_head_dim_calculation(self, mha: multi_head.MultiHeadAttention) -> None:
        """Test MHA head dimension calculation."""
        # This tests actual calculation logic
        assert mha.head_dim == 32  # 128 / 4

    def test_mha_forward_basic(self, mha: multi_head.MultiHeadAttention) -> None:
        """Test basic forward pass."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 128)

        output = mha(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, 128)

        # Check causal masking is applied (later positions shouldn't attend to future)
        # This is hard to test directly without inspecting attention weights

    def test_mha_qkv_projections(self, mha: multi_head.MultiHeadAttention) -> None:
        """Test Q, K, V projection computation."""
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 128)

        # Get projections
        q, k, v = mha._compute_qkv_projections(x)

        # Check shapes
        assert q.shape == (batch_size, seq_len, 128)
        assert k.shape == (batch_size, seq_len, 128)
        assert v.shape == (batch_size, seq_len, 128)

    def test_mha_multihead_reshape(self, mha: multi_head.MultiHeadAttention) -> None:
        """Test reshaping to multi-head format."""
        batch_size, seq_len = 2, 8
        tensor = torch.randn(batch_size, seq_len, 128)

        multihead = mha._reshape_to_multihead(tensor, batch_size, seq_len)

        # Check shape: [batch, n_heads, seq, head_dim]
        assert multihead.shape == (batch_size, 4, seq_len, 32)

        # Test merge back
        merged = mha._merge_heads(multihead)
        assert merged.shape == (batch_size, seq_len, 128)

    def test_mha_deterministic(self, mha: multi_head.MultiHeadAttention) -> None:
        """Test MHA is deterministic when dropout=0."""
        x = torch.randn(1, 10, 128)

        out1 = mha(x)
        out2 = mha(x)

        testing.assert_close(out1, out2)


class TestGroupedQueryAttention:
    """Test grouped query attention (Llama style)."""

    @pytest.fixture
    def rope_module(self) -> core.PrecomputedRoPE:
        """Create RoPE module for GQA."""
        return core.PrecomputedRoPE(
            dim=16,  # head_dim for GQA with hidden_dim=128, num_heads=8
            theta=10000.0,
            max_seq_len=512,
        )

    @pytest.fixture
    def gqa(self, rope_module: core.PrecomputedRoPE) -> grouped_query.GroupedQueryAttention:
        """Create GroupedQueryAttention instance."""
        return grouped_query.GroupedQueryAttention(
            hidden_dim=128,
            num_heads=8,
            num_kv_heads=2,  # 4:1 ratio
            rope_module=rope_module,
            dropout=None,
            bias=False,
            is_causal=True,
        )

    def test_gqa_n_rep_calculation(self, gqa: grouped_query.GroupedQueryAttention) -> None:
        """Test GQA repetition factor calculation."""
        # This tests the key GQA logic - how many times to repeat KV heads
        assert gqa.n_rep == 4  # 8 query heads / 2 kv heads

    def test_gqa_forward_basic(self, gqa: grouped_query.GroupedQueryAttention) -> None:
        """Test basic forward pass."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 128)

        output = gqa(x)

        assert output.shape == (batch_size, seq_len, 128)

    def test_gqa_kv_repeat(self, gqa: grouped_query.GroupedQueryAttention) -> None:
        """Test key/value head repetition."""
        batch_size, seq_len = 1, 8
        # Create tensor with shape [batch, n_kv_heads, seq, head_dim]
        kv_heads = torch.randn(batch_size, 2, seq_len, 16)

        # Repeat for all query heads
        repeated = gqa._repeat_kv_heads(kv_heads)

        # Should now have 8 heads (2 * 4)
        assert repeated.shape == (batch_size, 8, seq_len, 16)

        # Check that values are actually repeated
        for i in range(2):  # For each original KV head
            for j in range(4):  # For each repetition
                testing.assert_close(repeated[:, i * 4 + j], kv_heads[:, i])

    def test_gqa_with_position_offset(self, gqa: grouped_query.GroupedQueryAttention) -> None:
        """Test GQA with position offset for KV caching."""
        batch_size = 1
        x1 = torch.randn(batch_size, 1, 128)  # First token
        x2 = torch.randn(batch_size, 1, 128)  # Second token

        # Process tokens at different positions
        out1 = gqa(x1, position_offset=0)
        out2 = gqa(x2, position_offset=1)

        assert out1.shape == (batch_size, 1, 128)
        assert out2.shape == (batch_size, 1, 128)

    def test_gqa_with_kv_cache(self, gqa: grouped_query.GroupedQueryAttention) -> None:
        """Test GQA with static KV cache."""
        batch_size = 1

        # Install cache
        gqa.cache = kv_cache.KVCache(
            batch_size=batch_size,
            max_seq_length=100,
            n_kv_heads=2,
            head_dim=16,
            dtype=torch.float32,
            device="cpu",
        )

        # Process first token
        x1 = torch.randn(batch_size, 1, 128)
        out1 = gqa(x1, position_offset=0)

        # Check cache was updated
        assert not torch.all(gqa.cache.cache_k == 0)
        assert not torch.all(gqa.cache.cache_v == 0)

        # Process second token
        x2 = torch.randn(batch_size, 1, 128)
        out2 = gqa(x2, position_offset=1)

        assert out1.shape == (batch_size, 1, 128)
        assert out2.shape == (batch_size, 1, 128)


class TestMultiHeadLatentAttention:
    """Test multi-head latent attention (DeepSeek style)."""

    @pytest.fixture
    def partial_rope(self) -> rope_partial.PartialRoPE:
        """Create PartialRoPE for MLA."""
        return rope_partial.PartialRoPE(dim=64, theta=10000.0)

    @pytest.fixture
    def mla(self, partial_rope: rope_partial.PartialRoPE) -> multi_latent.MultiHeadLatentAttention:
        """Create MultiHeadLatentAttention instance."""
        return multi_latent.MultiHeadLatentAttention(
            hidden_dim=512,
            num_heads=8,
            head_dim=128,
            kv_compression_dim=256,
            query_compression_dim=256,
            rope_module=partial_rope,
            rope_dim=64,
            dropout=None,
            is_causal=True,
        )

    def test_mla_initialization(self, mla: multi_latent.MultiHeadLatentAttention) -> None:
        """Test MLA initializes correctly."""
        assert mla.hidden_dim == 512
        assert mla.num_heads == 8
        assert mla.head_dim == 128
        assert mla.rope_dim == 64

        # Check compression layers
        assert mla.kv_down.out_features == 256
        assert mla.query_down.out_features == 256

        # Check up-projections
        assert mla.key_up.out_features == 1024  # 8 * 128
        assert mla.value_up.out_features == 1024  # 8 * 128
        assert mla.query_up.out_features == 1024  # 8 * 128

        # Check RoPE projections
        assert mla.key_rope.out_features == 512  # 8 * 64
        assert mla.query_rope.out_features == 512  # 8 * 64

    def test_mla_forward_basic(self, mla: multi_latent.MultiHeadLatentAttention) -> None:
        """Test basic forward pass."""
        batch_size, seq_len = 1, 8
        x = torch.randn(batch_size, seq_len, 512)

        output = mla(x)

        assert output.shape == (batch_size, seq_len, 512)

    def test_mla_compression(self, mla: multi_latent.MultiHeadLatentAttention) -> None:
        """Test input compression."""
        batch_size, seq_len = 1, 8
        x = torch.randn(batch_size, seq_len, 512)

        kv_compressed, query_compressed = mla._compress_inputs(x)

        # Check compressed dimensions
        assert kv_compressed.shape == (batch_size, seq_len, 256)
        assert query_compressed.shape == (batch_size, seq_len, 256)

        # Verify compression actually reduces dimension
        assert kv_compressed.shape[-1] < x.shape[-1]

    def test_mla_rope_application(self, mla: multi_latent.MultiHeadLatentAttention) -> None:
        """Test RoPE is applied to position dimensions only."""
        batch_size, seq_len = 1, 4

        # Create rope tensor
        rope_tensor = torch.randn(batch_size, 8, seq_len, 64)

        # Apply RoPE
        rotated = mla._apply_rope_to_subset(rope_tensor)

        # Check shape is preserved
        assert rotated.shape == rope_tensor.shape

        # Check rotation was applied (output differs from input)
        assert not torch.allclose(rotated, rope_tensor)

    def test_mla_attention_scaling(self, mla: multi_latent.MultiHeadLatentAttention) -> None:
        """Test MLA uses correct attention scaling."""
        # MLA should scale by sqrt(head_dim + rope_dim)
        expected_scale = (128 + 64) ** 0.5

        # Create small tensors for testing
        q = torch.randn(1, 8, 4, 128 + 64)  # Total dim includes rope
        k = torch.randn(1, 8, 4, 128 + 64)

        # Compute scores
        scores = mla._compute_attention_scores(q, k)

        # Manually compute expected scores
        expected = torch.matmul(q, k.transpose(-2, -1)) / expected_scale

        testing.assert_close(scores, expected)


class TestAttentionMasking:
    """Test attention masking across different attention types."""

    def test_causal_mask_creation(self) -> None:
        """Test causal mask is created correctly."""
        mha = multi_head.MultiHeadAttention(
            hidden_dim=64,
            num_heads=4,
            is_causal=True,
        )

        mask = mha._create_causal_mask(seq_len=5, device=torch.device("cpu"))

        # Check shape
        assert mask.shape == (5, 5)

        # Check it's lower triangular
        expected = torch.tril(torch.ones(5, 5)).bool()
        testing.assert_close(mask, expected)

    @pytestmark_flash
    def test_attention_mask_application(self) -> None:
        """Test custom attention mask is applied correctly."""
        # This test only runs when Flash Attention is NOT available
        # because Flash Attention has limitations with custom masks
        mha = multi_head.MultiHeadAttention(
            hidden_dim=64,
            num_heads=4,
            is_causal=False,  # Disable causal to test custom mask
        )

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 64)

        # Create attention mask - mask out second half of first sequence
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, seq_len // 2 :] = 0

        # Forward pass
        output = mha(x, attention_mask=attention_mask)

        # Output should have correct shape
        assert output.shape == (batch_size, seq_len, 64)

    @pytestmark_no_flash
    def test_flash_attention_basic(self) -> None:
        """Test Flash Attention when available."""
        # This test only runs when Flash Attention IS available
        mha = multi_head.MultiHeadAttention(
            hidden_dim=64,
            num_heads=4,
            is_causal=True,
        )

        assert mha.use_flash_attention  # Verify flash is enabled

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 64)

        # Forward pass with Flash Attention
        output = mha(x)

        assert output.shape == (batch_size, seq_len, 64)
