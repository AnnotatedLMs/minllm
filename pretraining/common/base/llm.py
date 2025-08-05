# Standard Library
import abc
import typing

# Third Party
import jaxtyping
import torch

# Project
from pretraining.common.base import core
from pretraining.common.base import outputs


class BaseLLM(core.BaseTorchModule, abc.ABC):
    """
    Pure abstract base class for Language Models.

    This defines the minimal interface that all LLM implementations must follow.
    """

    # Subclasses must set these attributes in __init__
    vocab_size: int  # Vocabulary size of the model
    hidden_dim: int  # Hidden dimension (d_model) of the model
    n_layers: int  # Number of transformer layers
    block_size: int  # Maximum sequence length the model can handle

    @abc.abstractmethod
    def forward(
        self,
        input_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> outputs.ForwardOutput:
        """
        Unified forward pass for both training and inference.

        The model should use self.training to determine whether to:
        - Apply dropout (handled automatically by PyTorch)
        - Compute MTP logits (for models with multi-token prediction)
        - Collect auxiliary losses (for models with MoE)

        Args:
            input_ids: Input token indices
            attention_mask: Optional attention mask

        Returns:
            ForwardOutput containing:
                - logits: Main output logits
                - mtp_logits: Optional MTP logits (only during training)
                - aux_losses: Optional auxiliary losses (only during training)
        """
        pass

    @abc.abstractmethod
    def _compute_logits(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq vocab"]:
        """
        Compute output logits from hidden states.

        Args:
            hidden_states: Final hidden states from transformer

        Returns:
            Logits tensor for vocabulary prediction
        """
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def generate(
        self,
        idx: jaxtyping.Int[torch.Tensor, "batch seq"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: typing.Optional[int] = None,
        **kwargs,
    ) -> jaxtyping.Int[torch.Tensor, "batch new_seq"]:
        """
        Generate tokens autoregressively.

        This method should implement autoregressive generation by:
        1. Processing the input tokens through the model
        2. Sampling from the output distribution
        3. Appending the sampled token to the sequence
        4. Repeating until max_new_tokens are generated

        Different architectures may handle this differently:
        - GPT-2: Simple autoregressive generation with context window cropping
        - Llama: Can use KV cache for efficient generation
        - DeepSeek: May include MTP predictions in generation strategy

        Args:
            idx: Initial token indices [batch, seq]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = neutral, < 1.0 = less random, > 1.0 = more random)
            top_k: If set, only sample from top k most likely tokens
            **kwargs: Additional generation arguments (e.g., top_p, use_cache, attention_mask)

        Returns:
            Generated token indices [batch, seq + max_new_tokens]
        """
        pass
