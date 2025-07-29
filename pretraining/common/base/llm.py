# Standard Library
import abc
import typing

# Third Party
import jaxtyping
import torch

# Project
from pretraining.common.base import core
from pretraining.common.base import inputs
from pretraining.common.base import outputs
from pretraining.configs.model import initialization
from pretraining.configs.model.architectures import base as arch_base
from pretraining.utils import weight_init


class BaseLLM(core.BaseTorchModule, abc.ABC):
    """
    Pure abstract base class for Language Models.

    This defines the minimal interface that all LLM implementations must follow.
    """

    @abc.abstractmethod
    def training_forward(
        self,
        training_inputs: inputs.TrainingInputs,
    ) -> outputs.TrainingOutput:
        """
        Forward pass for training.

        Args:
            training_inputs: TrainingInputs containing:
                - input_ids: Input token indices
                - labels: Target tokens for loss computation
                - attention_mask: Optional attention mask
                - mtp_targets: Optional multi-token prediction targets (for DeepSeek3)

        Returns:
            TrainingOutput containing:
                - loss: Main cross-entropy loss
                - mtp_losses: Optional list of MTP losses (for DeepSeek3)
                - aux_losses: Optional list of auxiliary losses (e.g., from MoE)
        """
        pass

    @abc.abstractmethod
    def inference_forward(
        self,
        inference_inputs: inputs.InferenceInputs,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq vocab"]:
        """
        Forward pass for inference/generation.

        Args:
            inference_inputs: InferenceInputs containing:
                - input_ids: Input token indices
                - attention_mask: Optional attention mask
                - position_offset: Starting position for RoPE (for models with KV cache)

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
        config: arch_base.BaseLLMConfig,
    ) -> None:
        """
        Apply weight initialization based on config.

        This method should be called during __init__ by LLM implementations
        after all modules have been created.

        Args:
            config: The LLM configuration containing optional weight_init
        """
        if config.weight_init is None:
            # Use PyTorch defaults - nothing to do
            return

        if isinstance(config.weight_init, initialization.GPT2InitConfig):
            # Use TransformerWeightInitializer for GPT-2 style
            initializer = weight_init.TransformerWeightInitializer(
                n_layer=self.n_layers,
                std=config.weight_init.std,
                residual_pattern=config.weight_init.residual_pattern,
            )
            initializer.initialize(self)
        else:
            raise ValueError(
                f"Unknown weight initialization config type: {type(config.weight_init).__name__}. "
                f"Expected GPT2InitConfig or None."
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
            # Create InferenceInputs
            inference_inputs = inputs.InferenceInputs(
                input_ids=idx,
                attention_mask=kwargs.get("attention_mask"),
                position_offset=kwargs.get("position_offset", 0),
            )

            # Get logits from the model
            logits = self.inference_forward(inference_inputs)

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
