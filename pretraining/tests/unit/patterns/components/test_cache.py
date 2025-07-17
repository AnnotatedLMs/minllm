"""
Unit tests for KV cache component.

Tests static buffer-based KV caching for efficient generation.
"""

# Third Party
import pytest
import torch
import torch.testing

# Project
# Local
from pretraining.common.patterns.components import cache


class TestKVCache:
    """Test KVCache implementation."""

    @pytest.fixture
    def kv_cache(self) -> cache.KVCache:
        """Create a KV cache instance."""
        return cache.KVCache(
            batch_size=2,
            max_seq_length=100,
            n_kv_heads=4,
            head_dim=64,
            dtype=torch.float32,
            device="cpu",
        )

    def test_cache_initialization(self, kv_cache: cache.KVCache) -> None:
        """Test cache initializes with correct shapes."""
        # Check buffer shapes
        assert kv_cache.cache_k.shape == (2, 100, 4, 64)
        assert kv_cache.cache_v.shape == (2, 100, 4, 64)

        # Check buffers are initialized to zero
        assert torch.all(kv_cache.cache_k == 0)
        assert torch.all(kv_cache.cache_v == 0)

        # Check position tracking
        assert kv_cache.cache_position.item() == 0

    def test_cache_update_single_token(self, kv_cache: cache.KVCache) -> None:
        """Test updating cache with a single token."""
        # Create new K/V for position 0
        xk = torch.randn(2, 1, 4, 64)  # batch=2, seq=1, heads=4, dim=64
        xv = torch.randn(2, 1, 4, 64)

        # Update cache
        cached_k, cached_v = kv_cache.update(start_pos=0, xk=xk, xv=xv)

        # Check returned shapes (should include only the filled portion)
        assert cached_k.shape == (2, 1, 4, 64)
        assert cached_v.shape == (2, 1, 4, 64)

        # Check values were stored
        torch.testing.assert_close(cached_k, xk)
        torch.testing.assert_close(cached_v, xv)

        # Check internal buffer was updated
        torch.testing.assert_close(kv_cache.cache_k[:, 0:1], xk)
        torch.testing.assert_close(kv_cache.cache_v[:, 0:1], xv)

    def test_cache_update_sequence(self, kv_cache: cache.KVCache) -> None:
        """Test updating cache with a sequence of tokens."""
        # First update: positions 0-9
        xk1 = torch.randn(2, 10, 4, 64)
        xv1 = torch.randn(2, 10, 4, 64)
        cached_k1, cached_v1 = kv_cache.update(start_pos=0, xk=xk1, xv=xv1)

        assert cached_k1.shape == (2, 10, 4, 64)
        assert cached_v1.shape == (2, 10, 4, 64)

        # Second update: position 10 (single new token)
        xk2 = torch.randn(2, 1, 4, 64)
        xv2 = torch.randn(2, 1, 4, 64)
        cached_k2, cached_v2 = kv_cache.update(start_pos=10, xk=xk2, xv=xv2)

        # Should return all 11 positions
        assert cached_k2.shape == (2, 11, 4, 64)
        assert cached_v2.shape == (2, 11, 4, 64)

        # Check that old values are preserved
        torch.testing.assert_close(cached_k2[:, :10], xk1)
        torch.testing.assert_close(cached_v2[:, :10], xv1)

        # Check new values
        torch.testing.assert_close(cached_k2[:, 10:11], xk2)
        torch.testing.assert_close(cached_v2[:, 10:11], xv2)

    def test_cache_incremental_generation(self, kv_cache: cache.KVCache) -> None:
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

            cached_k, cached_v = kv_cache.update(start_pos=pos, xk=xk, xv=xv)

            # Check we get all tokens up to current position
            assert cached_k.shape == (2, pos + 1, 4, 64)
            assert cached_v.shape == (2, pos + 1, 4, 64)

            # Verify all previous tokens are correct
            expected_k = torch.cat(all_keys, dim=1)
            expected_v = torch.cat(all_values, dim=1)
            torch.testing.assert_close(cached_k, expected_k)
            torch.testing.assert_close(cached_v, expected_v)

    def test_cache_reset(self, kv_cache: cache.KVCache) -> None:
        """Test cache reset functionality."""
        # Fill cache with some data
        xk = torch.randn(2, 10, 4, 64)
        xv = torch.randn(2, 10, 4, 64)
        kv_cache.update(start_pos=0, xk=xk, xv=xv)

        # Verify data is stored
        assert not torch.all(kv_cache.cache_k == 0)
        assert not torch.all(kv_cache.cache_v == 0)

        # Reset cache
        kv_cache.reset()

        # Verify cache is cleared
        assert torch.all(kv_cache.cache_k == 0)
        assert torch.all(kv_cache.cache_v == 0)
        assert kv_cache.cache_position.item() == 0

    def test_cache_different_dtypes(self) -> None:
        """Test cache with different data types."""
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            kv_cache = cache.KVCache(
                batch_size=1,
                max_seq_length=10,
                n_kv_heads=2,
                head_dim=32,
                dtype=dtype,
                device="cpu",
            )

            # Check dtype is correct
            assert kv_cache.cache_k.dtype == dtype
            assert kv_cache.cache_v.dtype == dtype

            # Test update maintains dtype
            xk = torch.randn(1, 1, 2, 32, dtype=dtype)
            xv = torch.randn(1, 1, 2, 32, dtype=dtype)
            cached_k, cached_v = kv_cache.update(0, xk, xv)

            assert cached_k.dtype == dtype
            assert cached_v.dtype == dtype
