# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models import outputs


class AutoregressiveGenerationMixin:
    """
    Mixin for autoregressive text generation.

    Variation: Token-by-token generation with various sampling strategies
    Computation: Forward pass → sample → append → repeat
    Effect: Generates coherent text continuations
    """

    def _crop_context_to_block_size(
        self,
        token_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        block_size: int,
    ) -> jaxtyping.Int[torch.Tensor, "batch seq"]:
        """Crop context if it exceeds block_size."""
        if token_ids.size(1) <= block_size:
            return token_ids
        return token_ids[:, -block_size:]

    def _apply_temperature(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch vocab"],
        temperature: float,
    ) -> jaxtyping.Float[torch.Tensor, "batch vocab"]:
        """Apply temperature scaling to logits."""
        return logits / temperature

    def _apply_top_k_filtering(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch vocab"],
        top_k: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch vocab"]:
        """Apply top-k filtering to logits."""
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")
        return logits

    def _sample_from_logits(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch vocab"],
    ) -> jaxtyping.Int[torch.Tensor, "batch 1"]:
        """Sample token from logits distribution."""
        probs = nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _extract_last_token_logits(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
    ) -> jaxtyping.Float[torch.Tensor, "batch vocab"]:
        """Extract logits for the last token in sequence."""
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(
        self,
        model_forward: typing.Callable,
        block_size: int,
        token_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: typing.Optional[int] = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
    ) -> jaxtyping.Int[torch.Tensor, "batch new_seq"]:
        """
        Generate tokens autoregressively.

        Handles context window cropping when sequence exceeds block_size.
        """
        for _ in range(max_new_tokens):
            context_tokens = self._crop_context_to_block_size(token_ids, block_size)

            model_output: outputs.ForwardOutput = model_forward(
                input_ids=context_tokens,
                attention_mask=attention_mask,
            )
            logits = model_output.logits

            last_logits = self._extract_last_token_logits(logits)
            last_logits = self._apply_temperature(last_logits, temperature)

            if top_k is not None:
                last_logits = self._apply_top_k_filtering(last_logits, top_k)

            next_token = self._sample_from_logits(last_logits)
            token_ids = torch.cat((token_ids, next_token), dim=1)

        return token_ids


class CachedGenerationMixin:
    """
    Mixin for efficient cached generation using KV cache.

    Variation: Uses KV cache to avoid recomputing past attention states
    Computation: Only processes new tokens, reuses cached K/V for past tokens
    Effect: O(1) per token generation instead of O(n) for sequence length n
    """

    def _validate_sequence_length(
        self,
        seq_len: int,
        block_size: int,
    ) -> None:
        """Check if sequence length is within block size."""
        if seq_len > block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {block_size}")

    def _compute_max_seq_length(
        self,
        initial_seq_len: int,
        max_new_tokens: int,
        block_size: int,
    ) -> int:
        """Compute maximum sequence length for cache allocation."""
        return min(initial_seq_len + max_new_tokens, block_size)

    def _should_stop_generation(
        self,
        current_length: int,
        block_size: int,
    ) -> bool:
        """Check if generation should stop due to length constraints."""
        return current_length >= block_size

    def _apply_temperature(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch vocab"],
        temperature: float,
    ) -> jaxtyping.Float[torch.Tensor, "batch vocab"]:
        """Apply temperature scaling to logits."""
        return logits / temperature

    def _apply_top_k_filtering(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch vocab"],
        top_k: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch vocab"]:
        """Apply top-k filtering to logits."""
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")
        return logits

    def _sample_from_logits(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch vocab"],
    ) -> jaxtyping.Int[torch.Tensor, "batch 1"]:
        """Sample token from logits distribution."""
        probs = nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _extract_last_token_logits(
        self,
        logits: jaxtyping.Float[torch.Tensor, "batch seq vocab"],
    ) -> jaxtyping.Float[torch.Tensor, "batch vocab"]:
        """Extract logits for the last token in sequence."""
        return logits[:, -1, :]

    @torch.no_grad()
    def generate_with_cache(
        self,
        model_forward: typing.Callable,
        setup_cache_fn: typing.Callable,
        reset_cache_fn: typing.Callable,
        block_size: int,
        token_ids: jaxtyping.Int[torch.Tensor, "batch seq"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: typing.Optional[int] = None,
    ) -> jaxtyping.Int[torch.Tensor, "batch new_seq"]:
        """
        Generate tokens using KV cache for efficiency.

        The cached generation process:
        1. Setup KV cache for all attention layers
        2. Process initial prompt (fill cache)
        3. Generate new tokens one at a time (use cache)
        4. Reset cache when done
        """
        batch_size = token_ids.shape[0]
        initial_seq_len = token_ids.shape[1]
        device = token_ids.device

        # Validate sequence length
        self._validate_sequence_length(initial_seq_len, block_size)

        # Setup KV cache
        max_seq_length = self._compute_max_seq_length(initial_seq_len, max_new_tokens, block_size)
        # Get model dtype from first parameter
        model_dtype = next(self.parameters()).dtype
        setup_cache_fn(
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            device=device,
            dtype=model_dtype,
        )

        try:
            position_offset = 0
            model_output: outputs.ForwardOutput = model_forward(
                input_ids=token_ids,
                attention_mask=None,
                position_offset=position_offset,
            )
            logits = model_output.logits

            for idx in range(max_new_tokens):
                current_length = initial_seq_len + idx
                if self._should_stop_generation(current_length, block_size):
                    break

                last_logits = self._extract_last_token_logits(logits)
                last_logits = self._apply_temperature(last_logits, temperature)

                if top_k is not None:
                    last_logits = self._apply_top_k_filtering(last_logits, top_k)

                next_token = self._sample_from_logits(last_logits)
                token_ids = torch.cat((token_ids, next_token), dim=1)

                position_offset = current_length

                model_output: outputs.ForwardOutput = model_forward(
                    input_ids=next_token,
                    attention_mask=None,
                    position_offset=position_offset,
                )
                logits = model_output.logits

        finally:
            reset_cache_fn()

        return token_ids
