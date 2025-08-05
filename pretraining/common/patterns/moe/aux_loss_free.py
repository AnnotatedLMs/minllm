# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project
from pretraining.common.patterns.moe import core


class AuxLossFreeMoE(core.MoE):
    """
    DeepSeek-V3 MoE implementation - Auxiliary Loss Free Mixture of Experts.

    Used by: DeepSeek-V3

    Variation: Shared expert + Routed experts with centroid-based affinity
    Computation: Affinity scores via sigmoid(token · centroid), bias-adjusted for load balancing
    Effect: Eliminates auxiliary loss during training while maintaining expert balance

    Variation: Dynamic bias adjustment tracks expert load
    Computation: Bias terms updated based on actual vs expected expert utilization
    Effect: Natural load balancing without explicit loss terms

    Variation: Shared expert processes all tokens
    Computation: Weighted combination of shared (ratio) and routed experts (1-ratio)
    Effect: Ensures all tokens get minimum processing while specializing via routing
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        intermediate_dim: typing.Optional[int] = None,
        shared_expert_ratio: float = 0.1,
        dropout: typing.Optional[float] = None,
    ):
        if intermediate_dim is None:
            intermediate_dim = 4 * hidden_dim

        super().__init__(hidden_dim, num_experts, num_experts_per_token, intermediate_dim, dropout)

        self.shared_expert_ratio = shared_expert_ratio

        # Expert centroids for affinity computation (e_i in the paper)
        self.expert_centroids = nn.Parameter(torch.randn(num_experts, hidden_dim))

        # Bias terms for load balancing (b_i in the paper)
        self.gate_bias = nn.Parameter(torch.zeros(num_experts))

        # Shared expert that always processes all tokens
        self.shared_expert = self._create_expert(hidden_dim, intermediate_dim)

        # Routed experts
        self.experts = nn.ModuleList(
            [self._create_expert(hidden_dim, intermediate_dim) for _ in range(num_experts)]
        )

        # Load tracking for bias updates
        self.register_buffer("expert_load", torch.zeros(num_experts))
        self.bias_update_speed = 0.001

    def _create_expert(self, hidden_dim: int, intermediate_dim: int) -> nn.Module:
        """
        Create a single expert network for DeepSeek-V3.

        Uses standard FFN with GELU activation and optional dropout.
        """
        layers = [
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        ]

        if self.dropout is not None:
            layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)

    def _compute_centroid_affinity(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]:
        """
        Compute affinity scores between tokens and expert centroids.

        DeepSeek specific: s_i,t = Sigmoid(u_t^T * e_i)
        where u_t is the token representation and e_i is the expert centroid.
        """
        # Compute dot product between tokens and expert centroids
        # x: [batch, seq, d_model], expert_centroids: [num_experts, d_model]
        dot_products: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        dot_products = torch.matmul(x, self.expert_centroids.T)

        # Apply sigmoid activation
        affinity_scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        affinity_scores = F.sigmoid(dot_products)

        return affinity_scores

    def _normalize_expert_weights(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch seq_len k"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len k"]:
        """
        Normalize expert weights to sum to 1.

        Override: DeepSeek uses sum normalization instead of softmax.
        """
        # Sum normalization
        score_sum: jaxtyping.Float[torch.Tensor, "batch seq_len 1"]
        score_sum = scores.sum(dim=-1, keepdim=True)

        weights: jaxtyping.Float[torch.Tensor, "batch seq_len k"]
        weights = scores / (score_sum + 1e-6)

        return weights

    def _compute_expert_load(
        self,
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq_len k"],
    ) -> jaxtyping.Float[torch.Tensor, "num_experts"]:
        """Compute how many tokens are assigned to each expert."""
        # One-hot encode expert assignments
        one_hot: jaxtyping.Float[torch.Tensor, "batch seq k num_experts"]
        one_hot = F.one_hot(expert_indices, self.num_experts).float()

        # Sum across all dimensions to get total load per expert
        expert_load: jaxtyping.Float[torch.Tensor, "num_experts"]
        expert_load = one_hot.sum(dim=[0, 1, 2])

        return expert_load

    def _update_load_balancing_bias(
        self,
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq_len k"],
    ) -> None:
        """
        Update bias terms based on expert load (no auxiliary loss).

        DeepSeek specific: Online bias adjustment for load balancing.
        """
        # Count how many tokens go to each expert
        current_load: jaxtyping.Float[torch.Tensor, "num_experts"]
        current_load = self._compute_expert_load(expert_indices)

        with torch.no_grad():
            # Exponential moving average of load
            self.expert_load.lerp_(current_load, 0.1)

            # Compute load imbalance
            mean_load: torch.Tensor = self.expert_load.mean()
            load_imbalance: jaxtyping.Float[torch.Tensor, "num_experts"]
            load_imbalance = self.expert_load - mean_load

            # Adjust bias - negative for overloaded experts
            self.gate_bias.data -= self.bias_update_speed * load_imbalance

    def _apply_shared_expert(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply shared expert to all tokens.

        DeepSeek specific: Shared expert with scaling.
        """
        shared_output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        shared_output = self.shared_expert(x)

        # Scale by shared expert ratio
        scaled_output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        scaled_output = shared_output * self.shared_expert_ratio

        return scaled_output

    def _compute_auxiliary_loss(
        self,
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq_len k"],
        affinity_scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"],
    ) -> torch.Tensor:
        """Compute complementary sequence-wise auxiliary loss."""
        batch_size, seq_len, k = expert_indices.shape

        # Compute f_i: fraction of tokens assigned to each expert
        one_hot: jaxtyping.Float[torch.Tensor, "batch seq k num_experts"]
        one_hot = F.one_hot(expert_indices, self.num_experts).float()

        indicator: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        indicator = one_hot.sum(dim=2)

        f_i: jaxtyping.Float[torch.Tensor, "batch num_experts"]
        f_i = indicator.sum(dim=1) * (self.num_experts / (self.num_experts_per_token * seq_len))

        # Compute P_i: average probability assigned to each expert
        affinity_sum: jaxtyping.Float[torch.Tensor, "batch seq_len 1"]
        affinity_sum = affinity_scores.sum(dim=2, keepdim=True)

        s_prime: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        s_prime = affinity_scores / (affinity_sum + 1e-10)

        P_i: jaxtyping.Float[torch.Tensor, "batch num_experts"]
        P_i = s_prime.mean(dim=1)

        # Balance loss: sum(f_i * P_i)
        balance_loss: jaxtyping.Float[torch.Tensor, "batch"]
        balance_loss = (f_i * P_i).sum(dim=1)

        aux_loss: torch.Tensor = balance_loss.mean()

        return aux_loss

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """
        Apply auxiliary-loss-free MoE to input tokens.

        The DeepSeek-V3 MoE process:
        1. Shared expert - apply shared FFN to all tokens (always active)
        2. Compute affinity - calculate sigmoid(token · centroid) for each expert
        3. Apply dynamic bias - add learned bias terms to balance expert load
        4. Select top-k - choose k highest scoring experts per token
        5. Normalize weights - use sum normalization (not softmax) for selected experts
        6. Route and compute - apply selected experts with normalized weights
        7. Update bias - adjust bias terms based on actual vs expected expert usage
        8. Combine outputs - weighted sum of shared (ratio) and routed (1-ratio) outputs

        Note: No auxiliary loss is added - load balancing happens through dynamic bias adjustment.
        """
        shared_output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        shared_output = self._apply_shared_expert(x)

        # Compute affinity scores
        affinity_scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        affinity_scores = self._compute_centroid_affinity(x)

        # Add bias for routing only
        scores_with_bias: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        scores_with_bias = affinity_scores + self.gate_bias

        # Select top-k experts based on biased scores
        _, top_k_indices = self._select_top_k_experts(scores_with_bias, self.num_experts_per_token)

        # Extract affinity scores for selected experts (without bias)
        batch_size, seq_len, _ = x.shape
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).unsqueeze(2)
        seq_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(2)

        top_k_scores: jaxtyping.Float[torch.Tensor, "batch seq_len k"]
        top_k_scores = affinity_scores[batch_indices, seq_indices, top_k_indices]

        # Normalize weights
        expert_weights: jaxtyping.Float[torch.Tensor, "batch seq_len k"]
        expert_weights = self._normalize_expert_weights(top_k_scores)

        # Initialize routed output
        routed_output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        routed_output = torch.zeros_like(x)

        # Process top-k experts for each token
        for i in range(self.num_experts_per_token):
            # Get expert assignment for this position
            expert_idx_per_token: jaxtyping.Int[torch.Tensor, "batch seq"]
            expert_idx_per_token = top_k_indices[:, :, i]

            expert_weight_per_token: jaxtyping.Float[torch.Tensor, "batch seq"]
            expert_weight_per_token = expert_weights[:, :, i]

            # Process each expert
            for expert_idx in range(self.num_experts):
                # Find tokens assigned to this expert
                mask: jaxtyping.Bool[torch.Tensor, "batch seq"]
                mask = expert_idx_per_token == expert_idx

                if mask.any():
                    # Get tokens for this expert
                    expert_input: jaxtyping.Float[torch.Tensor, "num_tokens hidden_dim"]
                    expert_input = x[mask]

                    # Apply expert
                    expert_output: jaxtyping.Float[torch.Tensor, "num_tokens hidden_dim"]
                    expert_output = self.experts[expert_idx](expert_input)

                    # Scale by gating weight
                    weights: jaxtyping.Float[torch.Tensor, "num_tokens 1"]
                    weights = expert_weight_per_token[mask].unsqueeze(-1)

                    weighted_output: jaxtyping.Float[torch.Tensor, "num_tokens hidden_dim"]
                    weighted_output = expert_output * weights

                    # Add to output
                    routed_output[mask] += weighted_output

        if self.training:
            # Update load balancing bias
            self._update_load_balancing_bias(top_k_indices)

            # Compute auxiliary loss
            self._aux_loss = self._compute_auxiliary_loss(top_k_indices, affinity_scores)

        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = shared_output + routed_output

        return output
