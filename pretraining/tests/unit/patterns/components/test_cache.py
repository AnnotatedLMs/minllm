"""
Unit tests for KV cache component.

Tests static buffer-based KV caching for efficient generation.
"""

# Third Party
import pytest
import torch
from torch import nn
from torch import testing

# Project
from pretraining.common.models.attention import cache_mixins


class TestCacheModule(nn.Module, cache_mixins.CachedAttentionMixin):
    """Test module that includes the cache mixin."""

    def __init__(self):
        super().__init__()


class TestKVCache:
    """Test KVCache implementation via CachedAttentionMixin."""

    @pytest.fixture
    def cache(self) -> TestCacheModule:
        """Create a test module with cache capabilities."""
        module = TestCacheModule()
        module.setup_cache(
            batch_size=2,
            max_seq_length=100,
            n_kv_heads=4,
            head_dim=64,
            dtype=torch.float32,
            device="cpu",
        )
        return module

    def test_cache_initialization(self, cache: TestCacheModule) -> None:
        """Test cache initializes with correct shapes."""
        # Check buffer shapes
        assert cache.cache_k.shape == (2, 100, 4, 64)
        assert cache.cache_v.shape == (2, 100, 4, 64)

        # Check buffers are initialized to zero
        assert torch.all(cache.cache_k == 0)
        assert torch.all(cache.cache_v == 0)

        # Check position tracking
        assert cache.cache_position.item() == 0

    def test_cache_update_single_token(self, cache: TestCacheModule) -> None:
        """Test updating cache with a single token."""
        # Create new K/V for position 0
        xk = torch.randn(2, 1, 4, 64)  # batch=2, seq=1, heads=4, dim=64
        xv = torch.randn(2, 1, 4, 64)

        # Update cache
        cached_k, cached_v = cache.update_and_get_cached_kv(start_pos=0, xk=xk, xv=xv)

        # Check returned shapes (should include only the filled portion)
        assert cached_k.shape == (2, 1, 4, 64)
        assert cached_v.shape == (2, 1, 4, 64)

        # Check values were stored
        testing.assert_close(cached_k, xk)
        testing.assert_close(cached_v, xv)

        # Check internal buffer was updated
        testing.assert_close(cache.cache_k[:, 0:1], xk)
        testing.assert_close(cache.cache_v[:, 0:1], xv)

    def test_cache_update_sequence(self, cache: TestCacheModule) -> None:
        """Test updating cache with a sequence of tokens."""
        # First update: positions 0-9
        xk1 = torch.randn(2, 10, 4, 64)
        xv1 = torch.randn(2, 10, 4, 64)
        cached_k1, cached_v1 = cache.update_and_get_cached_kv(start_pos=0, xk=xk1, xv=xv1)

        assert cached_k1.shape == (2, 10, 4, 64)
        assert cached_v1.shape == (2, 10, 4, 64)

        # Second update: position 10 (single new token)
        xk2 = torch.randn(2, 1, 4, 64)
        xv2 = torch.randn(2, 1, 4, 64)
        cached_k2, cached_v2 = cache.update_and_get_cached_kv(start_pos=10, xk=xk2, xv=xv2)

        # Should return all 11 positions
        assert cached_k2.shape == (2, 11, 4, 64)
        assert cached_v2.shape == (2, 11, 4, 64)

        # Check that old values are preserved
        testing.assert_close(cached_k2[:, :10], xk1)
        testing.assert_close(cached_v2[:, :10], xv1)

        # Check new values
        testing.assert_close(cached_k2[:, 10:11], xk2)
        testing.assert_close(cached_v2[:, 10:11], xv2)

    def test_cache_incremental_generation(self, cache: TestCacheModule) -> None:
        """Test cache behavior during incremental token generation."""
        # Simulate generation process
        all_keys = []
        all_values = []

        # Generate 20 tokens one by one
        for pos in range(20):
            xk = torch.randn(2, 1, 4, 64)
            xv = torch.randn(2, 1, 4, 64)

            all_keys.append(xk)
            all_values.append(xv)

            cached_k, cached_v = cache.update_and_get_cached_kv(start_pos=pos, xk=xk, xv=xv)

            # Check we get all tokens up to current position
            assert cached_k.shape == (2, pos + 1, 4, 64)
            assert cached_v.shape == (2, pos + 1, 4, 64)

            # Verify all previous tokens are correct
            expected_k = torch.cat(all_keys, dim=1)
            expected_v = torch.cat(all_values, dim=1)
            testing.assert_close(cached_k, expected_k)
            testing.assert_close(cached_v, expected_v)

    def test_cache_reset(self, cache: TestCacheModule) -> None:
        """Test cache reset functionality."""
        # Fill cache with some data
        xk = torch.randn(2, 10, 4, 64)
        xv = torch.randn(2, 10, 4, 64)
        cache.update_and_get_cached_kv(start_pos=0, xk=xk, xv=xv)

        # Verify data is stored
        assert not torch.all(cache.cache_k == 0)
        assert not torch.all(cache.cache_v == 0)

        # Reset cache
        cache.reset_cache()

        # Verify cache is cleared
        assert torch.all(cache.cache_k == 0)
        assert torch.all(cache.cache_v == 0)
        assert cache.cache_position.item() == 0

    def test_cache_different_dtypes(self) -> None:
        """Test cache with different data types."""
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            cache_instance = TestCacheModule()
            cache_instance.setup_cache(
                batch_size=1,
                max_seq_length=10,
                n_kv_heads=2,
                head_dim=32,
                dtype=dtype,
                device="cpu",
            )

            # Check dtype is correct
            assert cache_instance.cache_k.dtype == dtype
            assert cache_instance.cache_v.dtype == dtype

            # Test update maintains dtype
            xk = torch.randn(1, 1, 2, 32, dtype=dtype)
            xv = torch.randn(1, 1, 2, 32, dtype=dtype)
            cached_k, cached_v = cache_instance.update_and_get_cached_kv(0, xk, xv)

            assert cached_k.dtype == dtype
            assert cached_v.dtype == dtype
