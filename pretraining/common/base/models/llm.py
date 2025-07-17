# Standard Library
import abc
import typing

# Third Party
import jaxtyping
import torch

# Project
# Local
from pretraining.common.base.models import core

# Avoid circular import by using TYPE_CHECKING
if typing.TYPE_CHECKING:
    # Project
    from pretraining.configs.llm import llm_configs


class BaseLLM(core.BaseTorchModule, abc.ABC):
    """
    Pure abstract base class for Language Models.

    This defines the minimal interface that all LLM implementations must follow.
    """

    @abc.abstractmethod
    def training_forward(
        self,
        idx: jaxtyping.Int[torch.Tensor, "batch seq"],
        targets: jaxtyping.Int[torch.Tensor, "batch seq"],
        **kwargs,
    ) -> typing.Union[
        torch.Tensor,
        typing.Tuple[torch.Tensor, jaxtyping.Float[torch.Tensor, "batch seq vocab"]],
        typing.Tuple[
            torch.Tensor,
            jaxtyping.Float[torch.Tensor, "batch seq vocab"],
            typing.Dict[str, typing.Any],
        ],
    ]:
        """
        Forward pass for training.

        Args:
            idx: Input token indices
            targets: Target tokens for loss computation
            **kwargs: Additional architecture-specific arguments such as:
                - attention_mask: Attention mask for the sequence
                - output_hidden_states: Whether to return all hidden states
                - return_logits: Whether to return logits in addition to loss
                - etc.

        Returns:
            One of:
            - Loss tensor (default)
            - Tuple of (loss, logits) (if return_logits=True)
            - Tuple of (loss, logits, extras_dict) (if extras needed)

        The extras_dict may contain:
            - hidden_states: All hidden states if requested
            - aux_losses: Auxiliary losses (e.g., from MoE)
            - mtp_logits: Multi-token prediction logits
            - etc.
        """
        pass

    @abc.abstractmethod
    def inference_forward(
        self,
        idx: jaxtyping.Int[torch.Tensor, "batch seq"],
        **kwargs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq vocab"]:
        """
        Forward pass for inference/generation.

        Args:
            idx: Input token indices
            **kwargs: Additional architecture-specific arguments such as:
                - attention_mask: Attention mask for the sequence
                - position_offset: Starting position for RoPE (for models with KV cache)
                - etc.

        Returns:
            Logits tensor of shape [batch, seq, vocab]
        """
        pass

    @abc.abstractmethod
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters in the model.

        Args:
            non_embedding: If True, exclude position embeddings from count

        Returns:
            Total number of parameters
        """
        pass

    @abc.abstractmethod
    def compute_loss(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
        targets: jaxtyping.Int[torch.Tensor, "batch seq"],
        ignore_index: int = -100,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute loss for training.

        Args:
            logits: Model output logits
            targets: Target token indices
            ignore_index: Index to ignore in loss computation
            **kwargs: Additional loss-specific arguments

        Returns:
            Scalar loss tensor
        """
        pass

    def _apply_weight_initialization(
        self,
        config: typing.Union[
            "llm_configs.GPT2Config", "llm_configs.LlamaConfig", "llm_configs.DeepSeekConfig"
        ],
    ) -> None:
        """
        Apply weight initialization based on config.

        This method should be called during __init__ by LLM implementations
        after all modules have been created.

        Args:
            config: The LLM configuration containing weight_init_config
        """
        # Project
        from pretraining.common.utils.initialization import weight_init

        if config.weight_init_config.strategy == "gpt2":
            # Use TransformerWeightInitializer for GPT-2 style
            initializer = weight_init.TransformerWeightInitializer(
                n_layer=self.n_layers,
                std=config.weight_init_config.std,
                residual_pattern=config.weight_init_config.residual_pattern,
            )
            initializer.initialize(self)
        elif config.weight_init_config.strategy == "pytorch_default":
            # Use standard PyTorch initialization
            # PyTorch defaults are already applied when modules are created
            pass
        else:
            raise ValueError(
                f"Unknown initialization strategy: {config.weight_init_config.strategy}"
            )

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

        Args:
            idx: Initial token indices [batch, seq]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = neutral, < 1.0 = less random, > 1.0 = more random)
            top_k: If set, only sample from top k most likely tokens
            **kwargs: Additional generation arguments (e.g., top_p, repetition_penalty)

        Returns:
            Generated token indices [batch, seq + max_new_tokens]
        """
        # This is a default implementation that can be overridden by subclasses
        for _ in range(max_new_tokens):
            # Get logits from the model
            logits = self.forward(idx, targets=None, **kwargs)

            # Handle different return types
            if isinstance(logits, tuple):
                logits = logits[0]

            # Focus on the last token logits
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
