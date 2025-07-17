"""
Unit tests for transformer block implementations.

Tests BiasedLNTransformerBlock and RMSNormTransformerBlock.
"""

# Third Party
import pytest
import torch
import torch.nn as nn
import torch.testing

# Project
# Local
from pretraining.common.patterns import attention
from pretraining.common.patterns import transformer
from pretraining.common.patterns.components import position
from pretraining.configs.transformer import position_configs


class TestBiasedLNTransformerBlock:
    """Test GPT-2 style transformer block with LayerNorm and bias."""

    @pytest.fixture
    def gpt2_block(self) -> transformer.BiasedLNTransformerBlock:
        """Create GPT-2 style transformer block."""
        return transformer.BiasedLNTransformerBlock(
            hidden_dim=128,
            num_heads=4,
            dropout=0.1,
            bias=True,
            max_seq_length=512,
            activation="gelu",
            norm_eps=1e-5,
            use_flash_attention=True,
        )

    def test_biased_ln_initialization(
        self, gpt2_block: transformer.BiasedLNTransformerBlock
    ) -> None:
        """Test block initializes correctly."""
        # Check layer norms
        assert isinstance(gpt2_block.ln_1, nn.LayerNorm)
        assert isinstance(gpt2_block.ln_2, nn.LayerNorm)
        assert gpt2_block.ln_1.normalized_shape == (128,)
        assert gpt2_block.ln_2.normalized_shape == (128,)

        # Check layer norm has bias (elementwise_affine includes both weight and bias)
        assert gpt2_block.ln_1.elementwise_affine
        assert gpt2_block.ln_2.bias is not None

        # Check attention
        assert isinstance(gpt2_block.attn, attention.MultiHeadAttention)
        assert gpt2_block.attn.num_heads == 4
        assert gpt2_block.attn.hidden_dim == 128

        # Check FFN
        assert hasattr(gpt2_block, "ffn")
        assert gpt2_block.ffn.hidden_dim == 128

    def test_biased_ln_forward(self, gpt2_block: transformer.BiasedLNTransformerBlock) -> None:
        """Test forward pass."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 128)

        output = gpt2_block(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, 128)

        # Output should be different from input (transformations applied)
        assert not torch.allclose(output, x)

    def test_biased_ln_residual_connections(
        self, gpt2_block: transformer.BiasedLNTransformerBlock
    ) -> None:
        """Test residual connections are properly applied."""
        x = torch.randn(1, 5, 128)

        # Forward pass
        output = gpt2_block(x)

        # With residual connections, output shouldn't be too far from input
        # (especially with small random weights at initialization)
        relative_change = torch.norm(output - x) / torch.norm(x)
        assert relative_change < 2.0  # Output within 2x of input magnitude

    def test_biased_ln_attention_mask(
        self, gpt2_block: transformer.BiasedLNTransformerBlock
    ) -> None:
        """Test attention mask is properly passed through."""
        # Force disable flash attention for this test
        gpt2_block.attn.flash = False

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 128)

        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 4:] = 0  # Mask out second half of first sequence

        output = gpt2_block(x, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, 128)

    def test_biased_ln_dropout_behavior(
        self, gpt2_block: transformer.BiasedLNTransformerBlock
    ) -> None:
        """Test dropout behavior in train vs eval mode."""
        x = torch.randn(1, 10, 128)

        # Training mode
        gpt2_block.train()
        out1 = gpt2_block(x)
        out2 = gpt2_block(x)

        # Outputs should differ due to dropout
        assert not torch.allclose(out1, out2)

        # Eval mode
        gpt2_block.eval()
        out3 = gpt2_block(x)
        out4 = gpt2_block(x)

        # Outputs should be identical
        torch.testing.assert_close(out3, out4)


class TestRMSNormTransformerBlock:
    """Test modern transformer block with RMSNorm (Llama/DeepSeek style)."""

    @pytest.fixture
    def rope_module(self) -> position.PrecomputedRoPE:
        """Create RoPE module for testing."""
        config = position_configs.RoPEConfig(theta=10000.0)
        return position.PrecomputedRoPE(
            dim=16,  # head_dim for hidden_dim=128, num_heads=8
            config=config,
            max_seq_len=512,
        )

    @pytest.fixture
    def llama_block(
        self, rope_module: position.PrecomputedRoPE
    ) -> transformer.RMSNormTransformerBlock:
        """Create Llama style transformer block."""
        return transformer.RMSNormTransformerBlock(
            hidden_dim=128,
            num_heads=8,
            num_kv_heads=2,  # GQA
            rope_module=rope_module,
            dropout=0.0,
            bias=False,
            norm_eps=1e-5,
            rope_dim=16,  # Match RoPE module dim
            activation="silu",
            ffn_dim_multiplier=2.0,
            multiple_of=64,
            use_flash_attention=True,
        )

    @pytest.fixture
    def deepseek_block(
        self, rope_module: position.PrecomputedRoPE
    ) -> transformer.RMSNormTransformerBlock:
        """Create DeepSeek style transformer block with MoE."""
        return transformer.RMSNormTransformerBlock(
            hidden_dim=128,
            num_heads=8,
            rope_module=rope_module,
            dropout=0.0,
            bias=False,
            use_moe=True,
            num_experts=4,
            num_experts_per_token=2,
            use_flash_attention=True,
        )

    def test_rms_norm_initialization_llama(
        self, llama_block: transformer.RMSNormTransformerBlock
    ) -> None:
        """Test Llama block initialization."""
        # Check RMSNorm layers
        assert isinstance(llama_block.input_layernorm, nn.RMSNorm)
        assert isinstance(llama_block.post_attention_layernorm, nn.RMSNorm)

        # Check GQA attention
        assert isinstance(llama_block.attention, attention.GroupedQueryAttention)
        assert llama_block.attention.num_heads == 8
        assert llama_block.attention.num_kv_heads == 2

        # Check SwiGLU FFN
        assert hasattr(llama_block, "ffn")
        assert not hasattr(llama_block, "moe")

    def test_rms_norm_initialization_deepseek(
        self, deepseek_block: transformer.RMSNormTransformerBlock
    ) -> None:
        """Test DeepSeek block initialization with MoE."""
        # Check MoE instead of FFN
        assert hasattr(deepseek_block, "moe")
        assert not hasattr(deepseek_block, "ffn")
        assert deepseek_block.moe.num_experts == 4

    def test_rms_norm_forward_llama(self, llama_block: transformer.RMSNormTransformerBlock) -> None:
        """Test Llama block forward pass."""
        batch_size, seq_len = 1, 8
        x = torch.randn(batch_size, seq_len, 128)

        output = llama_block(x)

        assert output.shape == (batch_size, seq_len, 128)

    def test_rms_norm_forward_with_position_offset(
        self, llama_block: transformer.RMSNormTransformerBlock
    ) -> None:
        """Test forward pass with position offset for KV caching."""
        batch_size = 1
        x = torch.randn(batch_size, 1, 128)  # Single token

        # Process at different positions
        out0 = llama_block(x, position_offset=0)
        out5 = llama_block(x, position_offset=5)

        # Both should produce valid outputs
        assert out0.shape == (batch_size, 1, 128)
        assert out5.shape == (batch_size, 1, 128)

        # Outputs should differ due to different position embeddings
        assert not torch.allclose(out0, out5)

    def test_rms_norm_normalization(self, llama_block: transformer.RMSNormTransformerBlock) -> None:
        """Test RMSNorm behavior."""
        x = torch.randn(1, 5, 128)

        # Get normalized output from first norm layer
        normed = llama_block.input_layernorm(x)

        # Check RMS is approximately 1
        rms = torch.sqrt(torch.mean(normed**2, dim=-1))
        expected_rms = torch.ones_like(rms)
        torch.testing.assert_close(rms, expected_rms, rtol=1e-3, atol=1e-3)

    def test_rms_norm_no_bias(self, llama_block: transformer.RMSNormTransformerBlock) -> None:
        """Test that Llama block has no bias in linear layers."""
        # Check attention projections
        assert llama_block.attention.q_proj.bias is None
        assert llama_block.attention.k_proj.bias is None
        assert llama_block.attention.v_proj.bias is None
        assert llama_block.attention.o_proj.bias is None

        # Check FFN layers
        if hasattr(llama_block, "ffn"):
            assert llama_block.ffn.gate_proj.bias is None
            assert llama_block.ffn.up_proj.bias is None
            assert llama_block.ffn.down_proj.bias is None

    def test_transformer_block_attention_types(self) -> None:
        """Test different attention configurations in RMSNorm blocks."""
        # MHA configuration (for comparison)
        mha_block = transformer.RMSNormTransformerBlock(
            hidden_dim=128,
            num_heads=8,
            num_kv_heads=None,  # Regular MHA
            rope_module=None,  # No RoPE
            dropout=0.0,
            bias=True,
        )

        assert isinstance(mha_block.attention, attention.MultiHeadAttention)

        x = torch.randn(1, 10, 128)
        output = mha_block(x)
        assert output.shape == x.shape


class TestTransformerBlockIntegration:
    """Test integration between transformer blocks and other components."""

    def test_blocks_in_sequence(self) -> None:
        """Test multiple blocks can be chained."""
        blocks = nn.ModuleList(
            [
                transformer.BiasedLNTransformerBlock(
                    hidden_dim=64,
                    num_heads=4,
                    dropout=0.0,
                )
                for _ in range(3)
            ]
        )

        x = torch.randn(1, 10, 64)

        # Pass through all blocks
        hidden = x
        for block in blocks:
            hidden = block(hidden)

        # Output should have same shape as input
        assert hidden.shape == x.shape

        # But different values
        assert not torch.allclose(hidden, x)

    def test_gradient_flow(self) -> None:
        """Test gradients flow through transformer block."""
        block = transformer.BiasedLNTransformerBlock(
            hidden_dim=64,
            num_heads=4,
            dropout=0.0,
        )

        x = torch.randn(1, 5, 64, requires_grad=True)
        output = block(x)

        # Create dummy loss
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert torch.any(x.grad != 0)

        # Check block parameters have gradients
        for param in block.parameters():
            if param.requires_grad:
                assert param.grad is not None
