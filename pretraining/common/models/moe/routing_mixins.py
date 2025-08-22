# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn
from torch.nn import functional as F


class CentroidRoutingMixin:
    """
    Mixin for centroid-based expert routing using learned expert representations.

    Scholarship:
    DeepSeekMoE, 2024, https://arxiv.org/pdf/2401.06066
    Deepseek-V2, 2024, https://arxiv.org/pdf/2405.04434
    Deepseek-V3, 2025, https://arxiv.org/pdf/2412.19437

    Significance:
    Enables fine-grained expert specialization by learning what each expert is good at.
    Centroids act as learnable "signatures" that attract tokens with similar patterns.
    Unlike hard routing, uses soft weighted combinations for smoother gradients.

    Init:
    The centroids are defined in AuxLossFreeMoE as:
        self.expert_centroids = nn.Parameter(torch.randn(num_experts, hidden_dim))

    Step-by-step control flow (_compute_centroid_affinity):
    1. Receive token embeddings of shape [batch, seq_len, hidden_dim]
    2. Compute dot product between each token and each expert centroid
    3. Apply sigmoid to get affinity scores between 0 and 1
    4. Return affinity matrix showing how well each token matches each expert

    Step-by-step control flow (_select_top_k_experts):
    1. Receive scores (usually biased for load balancing)
    2. Select k highest-scoring experts for each token
    3. Return both the top-k scores and their indices

    Step-by-step control flow (_normalize_expert_weights):
    1. Receive raw affinity scores for selected experts
    2. Sum the scores for each token
    3. Divide each score by the sum to get weights that sum to 1
    4. Return normalized weights for combining expert outputs

    Learning process:
    - Expert centroids (self.expert_centroids: nn.Parameter):
      - Learn to represent what kind of tokens each expert processes well
      - When a token is misclassified: loss increases, producing gradients
      - Gradients flow back through the sigmoid and dot product operations
      - If expert helped: centroid moves closer to tokens it processed (positive gradient)
      - If expert hurt: centroid moves away from tokens it processed (negative gradient)
      - Result: each centroid becomes a learned representation of its expert's specialty

    - Routing decisions:
      - Higher affinity scores lead to higher expert weights
      - Experts that reduce loss get stronger affinity to similar future tokens
      - Experts that increase loss get weaker affinity to similar tokens
      - Result: tokens naturally flow to experts that handle them well
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
