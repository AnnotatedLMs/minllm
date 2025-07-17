#!/usr/bin/env python3
"""Test Flash Attention with different dropout configurations."""

# Third Party
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from torch.nn.attention import sdpa_kernel

print("=== Testing Flash Attention + Dropout Theory ===\n")

# Create test tensors
batch_size, num_heads, seq_len, head_dim = 1, 4, 8, 32
device = "cpu"  # We're on CPU
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

print("Test 1: Flash Attention with NO dropout (should pass)")
try:
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,  # No dropout
            is_causal=True,
        )
    print("✓ PASSED - Flash Attention works with dropout=0.0")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"✗ FAILED - Error: {e}")

print("\nTest 2: Regular attention (no forcing) with dropout (should pass)")
try:
    # Don't force any backend - let PyTorch choose
    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.1,  # With dropout
        is_causal=True,
    )
    print("✓ PASSED - Regular attention works with dropout=0.1")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"✗ FAILED - Error: {e}")

print("\nTest 3: Flash Attention with dropout (should fail)")
try:
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.1,  # With dropout
            is_causal=True,
        )
    print("✗ UNEXPECTED - Flash Attention should NOT work with dropout>0")
    print(f"  Output shape: {output.shape}")
except RuntimeError as e:
    print(f"✓ EXPECTED FAILURE - Error: {e}")
    print("  This confirms Flash Attention doesn't support dropout")

print("\nTest 4: Math backend with dropout (should pass)")
try:
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.1,  # With dropout
            is_causal=True,
        )
    print("✓ PASSED - Math backend works with dropout=0.1")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"✗ FAILED - Error: {e}")

print("\n=== Summary ===")
print("Our theory is CORRECT if:")
print("- Test 1 (Flash, no dropout): PASSES")
print("- Test 2 (Auto, with dropout): PASSES")
print("- Test 3 (Flash, with dropout): FAILS")
print("- Test 4 (Math, with dropout): PASSES")
