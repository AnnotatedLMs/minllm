# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project
from pretraining.common.patterns.moe import base


class AuxLossFreeMoE(base.MoE):
    """
    Auxiliary-loss-free MoE pattern.

    Used by: DeepSeek-V3.
    Pattern: Shared expert + Routed experts + Dynamic bias adjustment

    Uses bias terms and online updates instead of auxiliary loss for load balancing.
    Includes a shared expert that processes all tokens to maintain baseline capacity.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        intermediate_dim: typing.Optional[int] = None,
        shared_expert_ratio: float = 0.1,
        dropout: float = 0.0,
    ):
        if intermediate_dim is None:
            intermediate_dim = 4 * hidden_dim

        super().__init__(hidden_dim, num_experts, num_experts_per_token, intermediate_dim, dropout)

        self.shared_expert_ratio = shared_expert_ratio
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
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

    def _compute_gating_with_bias(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq num_experts"]:
        """
        Compute gating scores with load balancing bias.

        Override: DeepSeek adds bias and uses sigmoid activation.
        """
        # Linear gating scores
        scores: jaxtyping.Float[torch.Tensor, "batch seq num_experts"]
        scores = self._compute_gating_scores(x)

        # Add load balancing bias
        scores_with_bias: jaxtyping.Float[torch.Tensor, "batch seq num_experts"]
        scores_with_bias = scores + self.gate_bias

        # Apply sigmoid for stable gradients
        gating_scores: jaxtyping.Float[torch.Tensor, "batch seq num_experts"]
        gating_scores = F.sigmoid(scores_with_bias)

        return gating_scores

    def _normalize_expert_weights(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch seq k"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq k"]:
        """
        Normalize expert weights to sum to 1.

        Override: DeepSeek uses sum normalization instead of softmax.
        """
        # Sum normalization
        score_sum: jaxtyping.Float[torch.Tensor, "batch seq 1"]
        score_sum = scores.sum(dim=-1, keepdim=True)

        weights: jaxtyping.Float[torch.Tensor, "batch seq k"]
        weights = scores / (score_sum + 1e-6)

        return weights

    def _compute_expert_load(
        self,
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq k"],
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
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq k"],
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
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply shared expert to all tokens.

        DeepSeek specific: Shared expert with scaling.
        """
        shared_output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        shared_output = self.shared_expert(x)

        # Scale by shared expert ratio
        scaled_output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        scaled_output = shared_output * self.shared_expert_ratio

        return scaled_output

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply auxiliary-loss-free MoE.

        The process:
        1. Apply shared expert to all tokens
        2. Compute gating scores with dynamic bias
        3. Select top-k experts per token
        4. Normalize weights using sum normalization
        5. Route tokens to selected experts
        6. Update load balancing bias (no aux loss)
        7. Combine shared and routed expert outputs
        """
        shared_output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        shared_output = self._apply_shared_expert(x)

        scores: jaxtyping.Float[torch.Tensor, "batch seq num_experts"]
        scores = self._compute_gating_with_bias(x)

        top_k_scores: jaxtyping.Float[torch.Tensor, "batch seq k"]
        top_k_indices: jaxtyping.Int[torch.Tensor, "batch seq k"]
        top_k_scores, top_k_indices = self._select_top_k_experts(scores, self.num_experts_per_token)

        expert_weights: jaxtyping.Float[torch.Tensor, "batch seq k"]
        expert_weights = self._normalize_expert_weights(top_k_scores)

        # Initialize routed output
        routed_output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
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
                    expert_input: jaxtyping.Float[torch.Tensor, "num_tokens d_model"]
                    expert_input = x[mask]

                    # Apply expert
                    expert_output: jaxtyping.Float[torch.Tensor, "num_tokens d_model"]
                    expert_output = self.experts[expert_idx](expert_input)

                    # Scale by gating weight
                    weights: jaxtyping.Float[torch.Tensor, "num_tokens 1"]
                    weights = expert_weight_per_token[mask].unsqueeze(-1)

                    weighted_output: jaxtyping.Float[torch.Tensor, "num_tokens d_model"]
                    weighted_output = expert_output * weights

                    # Add to output
                    routed_output[mask] += weighted_output

        if self.training:
            self._update_load_balancing_bias(top_k_indices)

        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = shared_output + routed_output

        return output
