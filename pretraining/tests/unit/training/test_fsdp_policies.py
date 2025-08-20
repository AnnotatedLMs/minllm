# Standard Library
import pathlib
import unittest

# Third Party
import torch

# Project
from pretraining.common.models.architectures import deepseek3
from pretraining.common.models.architectures import gpt2
from pretraining.common.models.architectures import llama3
from pretraining.common.models.blocks import block_group
from pretraining.common.models.blocks import deepseek3_blocks
from pretraining.common.models.blocks import gpt2_blocks
from pretraining.common.models.blocks import llama3_blocks
from pretraining.configs import loader
from pretraining.configs.model.architectures import deepseek
from pretraining.configs.model.architectures import gpt
from pretraining.configs.model.architectures import llama
from pretraining.configs.training import execution_configs


def get_debug_configs_dir() -> pathlib.Path:
    """Get the path to debug configs directory."""
    # Find project root by looking for pyproject.toml
    current = pathlib.Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current / "pretraining" / "configs" / "examples" / "debug"
        current = current.parent
    raise RuntimeError("Could not find project root")


class TestFSDPWrapPolicies(unittest.TestCase):
    """Test FSDP wrap policies for different architectures."""

    def setUp(self):
        """Set up test configurations."""
        # Load configs from the debug YAML files
        debug_dir = get_debug_configs_dir()

        # Load GPT-2 config
        gpt2_training = loader.load_training_config(
            debug_dir / "gpt2_debug_cpu.yaml", gpt.GPT2Config
        )
        self.gpt2_config = gpt2_training.llm

        # Load Llama3 config
        llama_training = loader.load_training_config(
            debug_dir / "llama3_debug_cpu.yaml", llama.Llama3Config
        )
        self.llama_config = llama_training.llm

        # Load DeepSeek3 config
        deepseek_training = loader.load_training_config(
            debug_dir / "deepseek3_debug_cpu.yaml", deepseek.DeepSeek3Config
        )
        self.deepseek_config = deepseek_training.llm

    def test_gpt2_fsdp_methods(self):
        """Test GPT2 FSDP methods."""
        model = gpt2.GPT2.from_config(self.gpt2_config)

        # Test get_fsdp_wrappable_modules
        wrappable = model.get_fsdp_wrappable_modules()
        self.assertIn(gpt2_blocks.GPT2TransformerBlock, wrappable)

        # Test get_transformer_blocks
        blocks = model.get_transformer_blocks()
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(isinstance(b, gpt2_blocks.GPT2TransformerBlock) for b in blocks))

        # Test get_fsdp_special_modules
        special = model.get_fsdp_special_modules()
        # Should include token_embeddings and lm_head (weight-tied)
        self.assertGreaterEqual(len(special), 1)

    def test_llama3_fsdp_methods(self):
        """Test Llama3 FSDP methods."""
        model = llama3.Llama3.from_config(self.llama_config)

        # Test get_fsdp_wrappable_modules
        wrappable = model.get_fsdp_wrappable_modules()
        self.assertIn(llama3_blocks.Llama3TransformerBlock, wrappable)

        # Test get_transformer_blocks
        blocks = model.get_transformer_blocks()
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(isinstance(b, llama3_blocks.Llama3TransformerBlock) for b in blocks))

        # Test get_fsdp_special_modules
        special = model.get_fsdp_special_modules()
        # Should include token_embeddings and lm_head
        self.assertGreaterEqual(len(special), 2)

    def test_deepseek3_fsdp_methods(self):
        """Test DeepSeek3 FSDP methods."""
        model = deepseek3.DeepSeek3.from_config(self.deepseek_config)

        # Test get_fsdp_wrappable_modules
        wrappable = model.get_fsdp_wrappable_modules()
        self.assertIn(deepseek3_blocks.DeepSeek3TransformerBlock, wrappable)

        # Test get_transformer_blocks
        blocks = model.get_transformer_blocks()
        self.assertEqual(len(blocks), 2)
        self.assertTrue(
            all(isinstance(b, deepseek3_blocks.DeepSeek3TransformerBlock) for b in blocks)
        )

        # Test get_fsdp_special_modules
        special = model.get_fsdp_special_modules()
        # Should include token_embeddings, lm_head, and MTP heads
        self.assertGreaterEqual(len(special), 2)

    def test_gpt2_wrap_policy_by_block(self):
        """Test GPT2 wrap policy with BY_BLOCK strategy."""
        model = gpt2.GPT2.from_config(self.gpt2_config)
        # Pass the enum member directly, not through config
        wrap_policy = model.get_fsdp_wrap_policy(execution_configs.FSDPWrapStrategy.BY_BLOCK)
        self.assertIsNotNone(wrap_policy)

        # Test that blocks are wrapped
        for block in model.get_transformer_blocks():
            # In BY_BLOCK mode, the policy should return True for blocks
            should_wrap = wrap_policy(block, recurse=False, nonwrapped_numel=0)
            self.assertTrue(should_wrap)

        # Test that embeddings are not wrapped in BY_BLOCK mode
        should_wrap = wrap_policy(model.token_embeddings, recurse=False, nonwrapped_numel=0)
        self.assertFalse(should_wrap)

    def test_gpt2_wrap_policy_by_block_and_size(self):
        """Test GPT2 wrap policy with BY_BLOCK_AND_SIZE strategy."""
        model = gpt2.GPT2.from_config(self.gpt2_config)
        # Pass the enum member directly, not through config
        wrap_policy = model.get_fsdp_wrap_policy(
            execution_configs.FSDPWrapStrategy.BY_BLOCK_AND_SIZE
        )
        self.assertIsNotNone(wrap_policy)

        # Test that blocks are wrapped
        for block in model.get_transformer_blocks():
            should_wrap = wrap_policy(block, recurse=False, nonwrapped_numel=0)
            self.assertTrue(should_wrap)

        # Test that special modules are wrapped in BY_BLOCK_AND_SIZE mode
        should_wrap = wrap_policy(model.token_embeddings, recurse=False, nonwrapped_numel=0)
        self.assertTrue(should_wrap)

    def test_size_based_wrap_policy(self):
        """Test size-based wrap policy."""
        model = gpt2.GPT2.from_config(self.gpt2_config)
        # Pass the enum member directly
        wrap_policy = model.get_fsdp_wrap_policy(execution_configs.FSDPWrapStrategy.SIZE_BASED)
        # SIZE_BASED should return None, to be handled by PyTorch's auto wrap
        self.assertIsNone(wrap_policy)


class TestBlockGroup(unittest.TestCase):
    """Test BlockGroup functionality for FSDP."""

    def setUp(self):
        """Set up test configuration."""
        # Load GPT-2 config from debug YAML
        debug_dir = get_debug_configs_dir()
        gpt2_training = loader.load_training_config(
            debug_dir / "gpt2_debug_cpu.yaml", gpt.GPT2Config
        )
        self.config = gpt2_training.llm

    def test_block_group_forward(self):
        """Test BlockGroup forward pass."""
        model = gpt2.GPT2.from_config(self.config)
        blocks = model.get_transformer_blocks()[:2]

        # Create a block group with first 2 blocks
        group = block_group.BlockGroup(blocks)

        # Test forward pass
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, self.config.transformer.hidden_dim)

        output = group(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
