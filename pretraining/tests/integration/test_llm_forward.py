"""
Integration tests for LLM forward methods.

Tests training_forward, inference_forward, and generate methods
across different architectures.
"""

# Third Party
import pytest
import torch
import torch.testing

# Project
from pretraining.common.patterns.architectures import deepseek3
from pretraining.common.patterns.architectures import gpt2
from pretraining.common.patterns.architectures import llama3
from pretraining.configs import loader
from pretraining.configs.model.architectures import deepseek
from pretraining.configs.model.architectures import gpt
from pretraining.configs.model.architectures import llama


class TestLLMForwardMethods:
    """Test forward methods across different LLM architectures."""

    @pytest.fixture
    def gpt2_model(self) -> gpt2.GPT2LLM:
        """Create a small GPT-2 model for testing."""
        config_path = "pretraining/configs/examples/debug/gpt2_debug.yaml"
        config = loader.load_training_config(config_path, gpt.GPT2Config)
        return gpt2.GPT2LLM(config.llm)

    @pytest.fixture
    def llama_model(self) -> llama3.LlamaLLM:
        """Create a small Llama model for testing."""
        config_path = "pretraining/configs/examples/debug/llama31_debug.yaml"
        config = loader.load_training_config(config_path, llama.Llama3Config)
        return llama3.LlamaLLM(config.llm)

    @pytest.fixture
    def deepseek_model(self) -> deepseek3.DeepSeekLLM:
        """Create a small DeepSeek model for testing."""
        config_path = "pretraining/configs/examples/debug/deepseek3_debug.yaml"
        config = loader.load_training_config(config_path, deepseek.DeepSeek3Config)
        return deepseek3.DeepSeekLLM(config.llm)

    def test_training_forward_gpt2(self, gpt2_model: gpt2.GPT2LLM) -> None:
        """Test GPT-2 training forward pass."""
        batch_size, seq_len = 2, 32
        vocab_size = gpt2_model.vocab_size

        # Create random inputs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass - should return just loss by default
        loss = gpt2_model.training_forward(input_ids, targets)

        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert loss.item() > 0  # Should be positive

        # Test with return_logits=True
        loss, logits = gpt2_model.training_forward(input_ids, targets, return_logits=True)

        assert isinstance(loss, torch.Tensor)
        assert logits.shape == (batch_size, seq_len, vocab_size)

        # Test with output_hidden_states=True
        loss, logits, extras = gpt2_model.training_forward(
            input_ids, targets, return_logits=True, output_hidden_states=True
        )

        assert "hidden_states" in extras
        assert len(extras["hidden_states"]) == gpt2_model.n_layers + 1  # +1 for final norm

    def test_inference_forward_gpt2(self, gpt2_model: gpt2.GPT2LLM) -> None:
        """Test GPT-2 inference forward pass."""
        batch_size, seq_len = 2, 32
        vocab_size = gpt2_model.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Inference forward should have no_grad decorator
        logits = gpt2_model.inference_forward(input_ids)

        # Check output
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert not logits.requires_grad  # Due to @torch.no_grad()

    def test_training_forward_llama(self, llama_model: llama3.LlamaLLM) -> None:
        """Test Llama training forward pass."""
        batch_size, seq_len = 1, 16  # Smaller for memory
        vocab_size = llama_model.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Test basic forward
        loss = llama_model.training_forward(input_ids, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_inference_forward_with_kv_cache(self, llama_model: llama3.LlamaLLM) -> None:
        """Test Llama inference with KV cache."""
        batch_size = 1
        vocab_size = llama_model.vocab_size

        # Install KV cache
        llama_model.install_kv_cache(
            batch_size=batch_size,
            max_seq_length=128,
            dtype=torch.float32,
            device="cpu",
        )

        # First token
        input_ids = torch.randint(0, vocab_size, (batch_size, 1))
        logits1 = llama_model.inference_forward(input_ids, position_offset=0)
        assert logits1.shape == (batch_size, 1, vocab_size)

        # Second token with position offset
        input_ids2 = torch.randint(0, vocab_size, (batch_size, 1))
        logits2 = llama_model.inference_forward(input_ids2, position_offset=1)
        assert logits2.shape == (batch_size, 1, vocab_size)

        # Clean up
        llama_model.clear_kv_cache()

    def test_training_forward_deepseek(self, deepseek_model: deepseek3.DeepSeekLLM) -> None:
        """Test DeepSeek training forward with MoE."""
        batch_size, seq_len = 1, 16
        vocab_size = deepseek_model.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        loss, logits, extras = deepseek_model.training_forward(
            input_ids, targets, return_logits=True
        )

        # Check outputs
        assert isinstance(loss, torch.Tensor)
        assert logits.shape == (batch_size, seq_len, vocab_size)

        # Check for MoE auxiliary losses
        if "aux_losses" in extras:
            assert isinstance(extras["aux_losses"], list)
            # Should have aux losses for each MoE layer
            assert len(extras["aux_losses"]) > 0

    def test_generate_gpt2(self, gpt2_model: gpt2.GPT2LLM) -> None:
        """Test GPT-2 generation."""
        batch_size = 1
        prompt_len = 5
        max_new_tokens = 10
        vocab_size = gpt2_model.vocab_size

        # Create prompt
        prompt = torch.randint(0, vocab_size, (batch_size, prompt_len))

        # Generate
        output = gpt2_model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
        )

        # Check output
        assert output.shape == (batch_size, prompt_len + max_new_tokens)
        assert torch.all(output[:, :prompt_len] == prompt)  # Prompt preserved

    def test_generate_llama_with_cache(self, llama_model: llama3.LlamaLLM) -> None:
        """Test Llama generation with static KV cache."""
        batch_size = 1
        prompt_len = 5
        max_new_tokens = 10
        vocab_size = llama_model.vocab_size

        # Create prompt
        prompt = torch.randint(0, vocab_size, (batch_size, prompt_len))

        # Generate with cache (default)
        output_with_cache = llama_model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            use_cache=True,
        )

        # Generate without cache
        output_no_cache = llama_model.generate(
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

    def test_attention_mask(self, gpt2_model: gpt2.GPT2LLM) -> None:
        """Test attention mask handling."""
        batch_size, seq_len = 2, 16
        vocab_size = gpt2_model.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create attention mask (mask out second half of second sequence)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[1, seq_len // 2 :] = 0

        # Forward pass
        logits = gpt2_model.inference_forward(input_ids, attention_mask=attention_mask)

        # Output should still have correct shape
        assert logits.shape == (batch_size, seq_len, vocab_size)
