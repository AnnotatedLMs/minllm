# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn
from torch.nn import functional as F


class CentroidRoutingMixin:
    """
    Mixin for Deepseek3 centroid-based expert routing.
    https://arxiv.org/pdf/2412.19437

    Significance:
    *Each* expert has a learned "centroid" vector representing what kind of tokens it specializes in.
    Tokens are routed to experts based on similarity to these centroids.

    Init:
    The centroids are defined in AuxLossFreeMoE as:
        self.expert_centroids = nn.Parameter(torch.randn(num_experts, hidden_dim))

    Routing approach:
    - One centroid vector per expert (shape: [num_experts, hidden_dim])
    - Centroids are nn.Parameters initialized randomly, learned via backprop
    - Affinity score = sigmoid(token · expert_centroid)
    - Higher affinity = token is more suited for that expert
    - "Soft" routing = experts get weighted contributions (not binary on/off)

    Step-by-step control flow:
    1. Token comes in with hidden_dim dimensions
    2. Compute affinity to ALL experts: sigmoid(token @ expert_centroid.T)
    3. Add bias to scores for load balancing (negative for overused experts, positive for underused)
    4. Select top-k experts based on BIASED scores
    5. Extract ORIGINAL affinity scores for those selected experts
    6. Normalize those scores to sum to 1 (these become the weights)
    7. Send token to each selected expert, multiply outputs by weights, sum them up

    Learning process:
    - Each expert's centroid learns through gradient descent from the language modeling loss
    - When an expert processes tokens well (low loss), its centroid gets updated to attract
      similar tokens in the future
    - When an expert processes tokens poorly (high loss), its centroid gets updated to repel
      similar tokens
    - Over time, each expert's centroid evolves to represent a specific type of content/pattern
      that the expert has learned to handle well

    Used by: DeepSeek-V3's AuxLossFreeMoE
    """

    def _compute_centroid_affinity(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        expert_centroids: nn.Parameter,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]:
        """Compute affinity scores between tokens and expert centroids."""
        # Compute dot product between tokens and expert centroids
        dot_products: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        dot_products = torch.matmul(x, expert_centroids.T)

        # Apply sigmoid activation for soft routing
        affinity_scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        affinity_scores = F.sigmoid(dot_products)

        return affinity_scores

    def _apply_routing_bias(
        self,
        affinity_scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"],
        gate_bias: nn.Parameter,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]:
        """Apply learnable bias for load balancing."""
        biased_scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        biased_scores = affinity_scores + gate_bias
        return biased_scores

    def _select_top_k_experts(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"],
        k: int,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq_len k"],
        jaxtyping.Int[torch.Tensor, "batch seq_len k"],
    ]:
        """Select top-k experts based on scores."""
        top_k_scores: jaxtyping.Float[torch.Tensor, "batch seq_len k"]
        top_k_indices: jaxtyping.Int[torch.Tensor, "batch seq_len k"]
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=-1)

        return top_k_scores, top_k_indices

    def _normalize_expert_weights(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts_per_token"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len num_experts_per_token"]:
        """Normalize expert weights to sum to 1."""
        # Sum normalization (not softmax)
        score_sum: jaxtyping.Float[torch.Tensor, "batch seq_len 1"]
        score_sum = scores.sum(dim=-1, keepdim=True)

        weights: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts_per_token"]
        weights = scores / (score_sum + 1e-6)

        return weights
