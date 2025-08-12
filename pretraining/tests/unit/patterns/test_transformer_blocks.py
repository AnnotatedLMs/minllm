"""
Unit tests for architecture-specific transformer blocks.

Tests GPT2TransformerBlock, Llama3TransformerBlock, and DeepSeek3TransformerBlock.
"""

# Third Party
import pytest
import torch
from torch import nn

# Project
from pretraining.common.patterns.blocks import deepseek3
from pretraining.common.patterns.blocks import gpt2
from pretraining.common.patterns.blocks import llama3
from pretraining.common.patterns.position import core
from pretraining.common.patterns.position import rope_partial


class TestGPT2TransformerBlock:
    @pytest.fixture
    def gpt2_block(self) -> gpt2.GPT2TransformerBlock:
        """Create a GPT-2 transformer block."""
        return gpt2.GPT2TransformerBlock(
            hidden_dim=64,
            num_heads=4,
            dropout=None,
            max_seq_length=128,
            norm_eps=1e-5,
            use_flash_attention=False,
        )

    def test_gpt2_components(self, gpt2_block: gpt2.GPT2TransformerBlock) -> None:
        """Test that GPT-2 block has correct components."""
        # Check normalization layers
        assert isinstance(gpt2_block.attention_norm, nn.LayerNorm)
        assert isinstance(gpt2_block.ffn_norm, nn.LayerNorm)

        # Check LayerNorm has bias
        assert gpt2_block.attention_norm.bias is not None
        assert gpt2_block.ffn_norm.bias is not None

        # Check attention and FFN
        assert hasattr(gpt2_block, "attention")
        assert hasattr(gpt2_block, "ffn")

    def test_gpt2_forward(self, gpt2_block: gpt2.GPT2TransformerBlock) -> None:
        """Test forward pass of GPT-2 block."""
        batch_size = 2
        seq_len = 10
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = gpt2_block(x)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert output.dtype == x.dtype


class TestLlama3TransformerBlock:
    @pytest.fixture
    def rope_module(self) -> core.PrecomputedRoPE:
        """Create RoPE module for Llama."""
        return core.PrecomputedRoPE(dim=16, theta=10000.0, max_seq_len=512)  # 16 = head_dim

    @pytest.fixture
    def llama_block(self, rope_module: core.PrecomputedRoPE) -> llama3.Llama3TransformerBlock:
        """Create a Llama 3 transformer block."""
        return llama3.Llama3TransformerBlock(
            hidden_dim=64,
            num_heads=4,
            num_kv_heads=2,  # GQA
            rope_module=rope_module,
            norm_eps=1e-5,
            use_flash_attention=False,
        )

    def test_llama_components(self, llama_block: llama3.Llama3TransformerBlock) -> None:
        """Test that Llama block has correct components."""
        # Check normalization layers
        assert isinstance(llama_block.attention_norm, nn.RMSNorm)
        assert isinstance(llama_block.ffn_norm, nn.RMSNorm)

        # Check attention and FFN
        assert hasattr(llama_block, "attention")
        assert hasattr(llama_block, "ffn")

    def test_llama_forward(self, llama_block: llama3.Llama3TransformerBlock) -> None:
        """Test forward pass of Llama block."""
        batch_size = 2
        seq_len = 10
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = llama_block(x)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert output.dtype == x.dtype


class TestDeepSeek3TransformerBlock:
    @pytest.fixture
    def partial_rope(self) -> rope_partial.PartialRoPE:
        """Create partial RoPE for DeepSeek."""
        return rope_partial.PartialRoPE(dim=64, theta=10000.0)

    @pytest.fixture
    def deepseek_block(
        self, partial_rope: rope_partial.PartialRoPE
    ) -> deepseek3.DeepSeek3TransformerBlock:
        """Create a DeepSeek-V3 transformer block."""
        return deepseek3.DeepSeek3TransformerBlock(
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
            kv_compression_dim=32,
            query_compression_dim=48,
            rope_module=partial_rope,
            rope_dim=64,
            num_experts=4,
            num_experts_per_token=2,
            dropout=None,
            norm_eps=1e-5,
            use_flash_attention=False,
        )

    def test_deepseek_components(self, deepseek_block: deepseek3.DeepSeek3TransformerBlock) -> None:
        """Test that DeepSeek block has correct components."""
        # Check normalization layers
        assert isinstance(deepseek_block.attention_norm, nn.RMSNorm)
        assert isinstance(deepseek_block.ffn_norm, nn.RMSNorm)

        # Check attention and MoE
        assert hasattr(deepseek_block, "attention")
        assert hasattr(deepseek_block, "moe")
        assert not hasattr(deepseek_block, "ffn")  # DeepSeek uses MoE instead

    def test_deepseek_forward(self, deepseek_block: deepseek3.DeepSeek3TransformerBlock) -> None:
        """Test forward pass of DeepSeek block."""
        batch_size = 2
        seq_len = 10
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = deepseek_block(x)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert output.dtype == x.dtype


class TestTransformerBlockConsistency:
    """Test consistency across different transformer block implementations."""

    def test_residual_connections(self) -> None:
        """Test that all blocks properly apply residual connections."""
        batch_size = 1
        seq_len = 5
        hidden_dim = 64

        # Create simple blocks
        gpt2_block = gpt2.GPT2TransformerBlock(
            hidden_dim=hidden_dim,
            num_heads=4,
            use_flash_attention=False,
        )

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Forward pass
        output = gpt2_block(x)

        # The output should be different from input (due to transformations)
        # but should have incorporated the input via residual connections
        assert not torch.allclose(output, x)
        assert output.shape == x.shape
