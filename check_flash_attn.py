#!/usr/bin/env python3
"""Check Flash Attention availability and capabilities."""

# Third Party
import torch
from torch.nn.attention import SDPBackend
from torch.nn.attention import sdpa_kernel

print("=== Flash Attention Support Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")

# Check if scaled_dot_product_attention exists
print(
    f"\nHas F.scaled_dot_product_attention: {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}"
)

# Test what backends are available
print("\n=== Testing Available Backends ===")

# Create test tensors
batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 64
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

# Test each backend
backends = [
    ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
    ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
    ("MATH", SDPBackend.MATH),
]

for name, backend in backends:
    try:
        with sdpa_kernel(backends=[backend]):
            # Test without dropout
            _ = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True
            )
        print(f"{name}: ✓ Available (no dropout)")

        # Test with dropout
        try:
            with sdpa_kernel(backends=[backend]):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.1, is_causal=True
                )
            print(f"{name}: ✓ Supports dropout")
        except RuntimeError as e:
            print(f"{name}: ✗ No dropout support - {str(e)}")

    except RuntimeError as e:
        print(f"{name}: ✗ Not available - {str(e)}")

print("\n=== Default Backend Selection ===")
# See what PyTorch chooses by default
try:
    # No dropout
    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    print("Default (no dropout): Success")

    # With dropout
    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.1, is_causal=True)
    print("Default (with dropout): Success")
except Exception as e:
    print(f"Default failed: {e}")

print("\n=== Your Machine Summary ===")
if device == "cpu":
    print("Running on CPU - Flash Attention requires CUDA GPU")
    print("Only MATH backend is available on CPU")
else:
    print("Running on CUDA GPU")
    print("Flash Attention may be available depending on GPU architecture")
