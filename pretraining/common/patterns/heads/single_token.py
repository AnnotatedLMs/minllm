# Third Party
import jaxtyping
import torch
import torch.nn as nn


class SingleTokenPredictionHead(nn.Module):
    """
    Standard language modeling head for next-token prediction.

    Used by: All autoregressive language models as their primary output
    - GPT-2: Uses weight tying by default
    - Llama: Always separate weights, no tying
    - DeepSeek: Separate weights + optional MTP as auxiliary task

    Variation: Weight tying vs separate weights
    - Weight tying: The output projection uses the SAME weight matrix as input embeddings
      (just transposed). If "cat" has embedding vector [0.1, 0.2, ...], then when
      predicting "cat", we use those same weights.
    - Separate weights: Output projection has its own weight matrix, independent of
      input embeddings. More parameters but more flexibility.

    Computation: hidden_states @ W^T + bias → logits over vocabulary
    - Input: [batch, seq, hidden_dim]
    - Weight: [vocab_size, hidden_dim]
    - Output: [batch, seq, vocab_size]

    Effect: Takes the model's internal understanding (hidden states) and converts it
    to scores for each possible next token in the vocabulary.

    Educational notes:
    - This is the primary output layer in all autoregressive language models
    - Even models with auxiliary tasks (like DeepSeek's MTP) still use this for main prediction
    - The output is "logits" (raw scores), not probabilities - softmax happens in the loss
    - Modern models (Llama, DeepSeek) avoid weight tying for training stability
    - Bias is typically false in modern models for efficiency

    The actual implementation is just nn.Linear, but we wrap it here for clarity
    and to document the pattern explicitly.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Just a linear projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size, bias=bias)

    def forward(
        self,
        hidden_states: jaxtyping.Float[torch.Tensor, "batch seq hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq vocab_size"]:
        """
        Project hidden states to vocabulary logits.

        The process:
        1. Linear transformation - maps from model's hidden dimension to vocabulary size
        2. No activation - raw logits are passed to loss function (CrossEntropy includes softmax)

        During training, these logits are compared against true next tokens.
        During generation, they're converted to probabilities for sampling.
        """
        logits = self.output_projection(hidden_states)
        return logits

    def tie_weights(self, embeddings: nn.Embedding) -> None:
        """
        Tie output projection weights to input embedding weights.

        What this means:
        - Normally: Input embeddings and output projection are two separate matrices
        - With tying: They share the SAME matrix (output uses transposed version)

        Why do this?
        - Saves parameters (one matrix instead of two)
        - Enforces symmetry: if "dog" and "puppy" have similar input embeddings,
          they'll also be predicted similarly

        Example:
        - Input embedding: token_id → embedding vector (lookup)
        - Output projection: hidden_state → token scores (matrix multiply)
        - With tying, both use the same underlying matrix

        Used by: GPT-2 (by default)
        Not used by: Llama, DeepSeek (they keep embeddings and output separate)
        """
        self.output_projection.weight = embeddings.weight
