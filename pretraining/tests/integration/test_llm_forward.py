"""
Integration tests for LLM forward methods.

Tests unified forward method and generate methods
across different architectures.
"""

# Third Party
import torch
import torch.testing

# Project
from pretraining.common.patterns.architectures import deepseek3
from pretraining.common.patterns.architectures import gpt2
from pretraining.common.patterns.architectures import llama3


class TestLLMForwardMethods:
    """Test forward methods across different LLM architectures."""

    def test_training_forward_gpt2(self, gpt2_debug_model: gpt2.GPT2) -> None:
        """Test GPT-2 training forward pass."""
        batch_size, seq_len = 2, 32
        vocab_size = gpt2_debug_model.vocab_size

        # Create random inputs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Put model in training mode
        gpt2_debug_model.train()

        # Forward pass - returns ForwardOutput
        output = gpt2_debug_model.forward(input_ids=input_ids)

        # Check logits
        assert hasattr(output, "logits")
        assert output.logits.shape == (batch_size, seq_len, vocab_size)
        # No MTP or aux losses for GPT2
        assert output.mtp_logits is None
        assert output.aux_losses is None

    def test_inference_forward_gpt2(self, gpt2_debug_model: gpt2.GPT2) -> None:
        """Test GPT-2 inference forward pass."""
        batch_size, seq_len = 2, 32
        vocab_size = gpt2_debug_model.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Put model in eval mode
        gpt2_debug_model.eval()

        # Forward pass in eval mode
        with torch.no_grad():
            output = gpt2_debug_model.forward(input_ids=input_ids)

        # Check output
        assert output.logits.shape == (batch_size, seq_len, vocab_size)
        assert not output.logits.requires_grad  # Due to torch.no_grad()

    def test_training_forward_llama(self, llama_debug_model: llama3.Llama3) -> None:
        """Test Llama training forward pass."""
        batch_size, seq_len = 1, 16  # Smaller for memory
        vocab_size = llama_debug_model.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Put model in training mode
        llama_debug_model.train()

        # Test basic forward
        output = llama_debug_model.forward(input_ids=input_ids)

        assert hasattr(output, "logits")
        assert output.logits.shape == (batch_size, seq_len, vocab_size)
        # No MTP or aux losses for Llama
        assert output.mtp_logits is None
        assert output.aux_losses is None

    def test_inference_forward_with_kv_cache(self, llama_debug_model: llama3.Llama3) -> None:
        """Test Llama inference with KV cache."""
        batch_size = 1
        vocab_size = llama_debug_model.vocab_size

        # Install KV cache
        llama_debug_model.install_kv_cache(
            batch_size=batch_size,
            max_seq_length=128,
            dtype=torch.float32,
            device="cpu",
        )

        # Put model in eval mode
        llama_debug_model.eval()

        # First token
        input_ids = torch.randint(0, vocab_size, (batch_size, 1))
        with torch.no_grad():
            output1 = llama_debug_model.forward(input_ids=input_ids, position_offset=0)
        assert output1.logits.shape == (batch_size, 1, vocab_size)

        # Second token with position offset
        input_ids2 = torch.randint(0, vocab_size, (batch_size, 1))
        with torch.no_grad():
            output2 = llama_debug_model.forward(input_ids=input_ids2, position_offset=1)
        assert output2.logits.shape == (batch_size, 1, vocab_size)

        # Clean up
        llama_debug_model.clear_kv_cache()

    def test_training_forward_deepseek(self, deepseek_debug_model: deepseek3.DeepSeek3) -> None:
        """Test DeepSeek training forward with MoE."""
        batch_size, seq_len = 1, 16
        vocab_size = deepseek_debug_model.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Put model in training mode
        deepseek_debug_model.train()

        # Forward pass
        output = deepseek_debug_model.forward(input_ids=input_ids)

        # Check outputs
        assert hasattr(output, "logits")
        assert output.logits.shape == (batch_size, seq_len, vocab_size)

        # DeepSeek has MTP logits in training mode
        assert output.mtp_logits is not None
        assert isinstance(output.mtp_logits, list)
        assert len(output.mtp_logits) > 0  # At least 1 MTP head
        # Check all MTP logits have correct shape
        for logit in output.mtp_logits:
            assert logit.shape == (batch_size, seq_len, vocab_size)

        # Check for MoE auxiliary losses
        if output.aux_losses is not None:
            assert isinstance(output.aux_losses, list)
            # Should have aux losses for each MoE layer
            assert len(output.aux_losses) > 0

    def test_generate_gpt2(self, gpt2_debug_model: gpt2.GPT2) -> None:
        """Test GPT-2 generation."""
        batch_size = 1
        prompt_len = 5
        max_new_tokens = 10
        vocab_size = gpt2_debug_model.vocab_size

        # Create prompt
        prompt = torch.randint(0, vocab_size, (batch_size, prompt_len))

        # Generate
        output = gpt2_debug_model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
        )

        # Check output
        assert output.shape == (batch_size, prompt_len + max_new_tokens)
        assert torch.all(output[:, :prompt_len] == prompt)  # Prompt preserved

    def test_generate_llama_with_cache(self, llama_debug_model: llama3.Llama3) -> None:
        """Test Llama generation with static KV cache."""
        batch_size = 1
        prompt_len = 5
        max_new_tokens = 10
        vocab_size = llama_debug_model.vocab_size

        # Create prompt
        prompt = torch.randint(0, vocab_size, (batch_size, prompt_len))

        # Generate with cache (default)
        output_with_cache = llama_debug_model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            use_cache=True,
        )

        # Generate without cache
        output_no_cache = llama_debug_model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            use_cache=False,
        )

        # Both should produce valid outputs
        assert output_with_cache.shape == (batch_size, prompt_len + max_new_tokens)
        assert output_no_cache.shape == (batch_size, prompt_len + max_new_tokens)

        # Prompts should be preserved
        assert torch.all(output_with_cache[:, :prompt_len] == prompt)
        assert torch.all(output_no_cache[:, :prompt_len] == prompt)

    def test_attention_mask(self, gpt2_debug_model: gpt2.GPT2) -> None:
        """Test attention mask handling."""
        batch_size, seq_len = 2, 16
        vocab_size = gpt2_debug_model.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create attention mask (mask out second half of second sequence)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[1, seq_len // 2 :] = 0

        # Put model in eval mode
        gpt2_debug_model.eval()

        # Forward pass with attention mask
        with torch.no_grad():
            output = gpt2_debug_model.forward(input_ids=input_ids, attention_mask=attention_mask)

        # Output should still have correct shape
        assert output.logits.shape == (batch_size, seq_len, vocab_size)
