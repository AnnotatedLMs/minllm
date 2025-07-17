# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project
# Local
from pretraining.common.base.models import moe

# TODO: why standard moe forward has method abstractions but not for aux loss-free?


class MoE(moe.BaseMoE):
    """
    Base class for MoE patterns with common implementations.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        intermediate_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout

        # For auxiliary loss tracking
        self._aux_loss = None

    def _create_expert(self, hidden_dim: int, intermediate_dim: int) -> nn.Module:
        """
        Create a single expert network.

        Standard implementation uses FFN with GELU activation.
        """
        return nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(self.dropout),
        )

    def _compute_gating_scores(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq num_experts"]:
        """
        Compute raw gating scores for each token-expert pair.

        Standard implementation uses linear projection through the learned gate.
        Each token gets a score for every expert.
        """
        scores: jaxtyping.Float[torch.Tensor, "batch seq num_experts"]
        scores = self.gate(x)
        return scores

    def _add_noise_for_exploration(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch seq num_experts"],
        noise_scale: float = 0.01,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq num_experts"]:
        """
        Add noise during training for exploration.

        Standard implementation adds Gaussian noise.
        """
        if self.training:
            noise: jaxtyping.Float[torch.Tensor, "batch seq num_experts"]
            noise = torch.randn_like(scores) * noise_scale

            noisy_scores: jaxtyping.Float[torch.Tensor, "batch seq num_experts"]
            noisy_scores = scores + noise

            return noisy_scores
        return scores

    def _select_top_k_experts(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch seq num_experts"],
        k: int,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq k"],
        jaxtyping.Int[torch.Tensor, "batch seq k"],
    ]:
        """
        Select top-k experts for each token.

        Standard implementation uses torch.topk.
        """
        top_k_scores: jaxtyping.Float[torch.Tensor, "batch seq k"]
        top_k_indices: jaxtyping.Int[torch.Tensor, "batch seq k"]
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=-1)

        return top_k_scores, top_k_indices

    def _normalize_expert_weights(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch seq k"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq k"]:
        """
        Normalize expert weights to sum to 1.

        Standard implementation uses softmax.
        """
        weights: jaxtyping.Float[torch.Tensor, "batch seq k"]
        weights = F.softmax(scores, dim=-1)
        return weights

    def get_auxiliary_loss(self) -> typing.Optional[torch.Tensor]:
        """Get auxiliary loss from last forward pass."""
        return self._aux_loss


class StandardMoE(MoE):
    """
    Standard Mixture of Experts with top-k routing and load balancing loss.

    Used by: Early MoE models, general pattern.
    Pattern: Gate → Top-k selection → Route → Weighted combination → Auxiliary loss

    This is the classic MoE pattern where tokens are routed to the top-k
    experts based on a learned gating function. Includes auxiliary loss
    for load balancing.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,  # top-k
        intermediate_dim: typing.Optional[int] = None,
        dropout: float = 0.0,
    ):
        if intermediate_dim is None:
            intermediate_dim = 4 * hidden_dim

        super().__init__(hidden_dim, num_experts, num_experts_per_token, intermediate_dim, dropout)

        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        # Experts - each is a standard FFN
        self.experts = nn.ModuleList(
            [self._create_expert(hidden_dim, intermediate_dim) for _ in range(num_experts)]
        )

    def _compute_load_balancing_loss(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch seq num_experts"],
        indices: jaxtyping.Int[torch.Tensor, "batch seq k"],
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for load balancing.

        This encourages uniform distribution of tokens across experts
        to ensure all experts are utilized effectively.
        """
        batch_size, seq_len, num_experts = scores.shape

        # Compute fraction of tokens routed to each expert
        # One-hot encode the selected experts
        one_hot_indices: jaxtyping.Float[torch.Tensor, "batch seq k num_experts"]
        one_hot_indices = F.one_hot(indices, num_experts).float()

        # Sum across selected positions to get expert assignment counts
        expert_mask: jaxtyping.Float[torch.Tensor, "batch seq num_experts"]
        expert_mask = one_hot_indices.sum(dim=2)

        # Average across batch and sequence to get load per expert
        tokens_per_expert: jaxtyping.Float[torch.Tensor, "num_experts"]
        tokens_per_expert = expert_mask.mean(dim=[0, 1])

        # Compute average probability assigned to each expert
        prob_per_expert: jaxtyping.Float[torch.Tensor, "num_experts"]
        prob_per_expert = F.softmax(scores, dim=-1).mean(dim=[0, 1])

        # Load balancing loss encourages uniform distribution
        # Scale by number of experts to make loss magnitude consistent
        aux_loss: torch.Tensor = num_experts * (tokens_per_expert * prob_per_expert).sum()

        return aux_loss

    def _find_expert_mask(
        self,
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq k"],
        expert_idx: int,
    ) -> jaxtyping.Bool[torch.Tensor, "batch seq"]:
        """Find which tokens are routed to a specific expert."""
        # Check all k positions for this expert
        expert_mask: jaxtyping.Bool[torch.Tensor, "batch seq"]
        expert_mask = (expert_indices == expert_idx).any(dim=-1)
        return expert_mask

    def _get_expert_weights(
        self,
        expert_weights: jaxtyping.Float[torch.Tensor, "batch seq k"],
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq k"],
        expert_idx: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq"]:
        """Get weights for tokens going to a specific expert."""
        # Find positions where this expert was selected
        k_positions: jaxtyping.Float[torch.Tensor, "batch seq k"]
        k_positions = (expert_indices == expert_idx).float()

        # Sum weights across k positions where this expert appears
        weights_for_expert: jaxtyping.Float[torch.Tensor, "batch seq"]
        weights_for_expert = (expert_weights * k_positions).sum(dim=-1)

        return weights_for_expert

    def _route_tokens_to_expert(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
        expert_idx: int,
        expert_mask: jaxtyping.Bool[torch.Tensor, "batch seq"],
    ) -> jaxtyping.Float[torch.Tensor, "num_tokens d_model"]:
        """Route tokens assigned to a specific expert."""
        # Extract tokens for this expert
        expert_input: jaxtyping.Float[torch.Tensor, "num_tokens d_model"]
        expert_input = x[expert_mask]

        # Apply expert
        expert_output: jaxtyping.Float[torch.Tensor, "num_tokens d_model"]
        expert_output = self.experts[expert_idx](expert_input)

        return expert_output

    def _combine_expert_outputs(
        self,
        expert_output: jaxtyping.Float[torch.Tensor, "num_tokens d_model"],
        expert_mask: jaxtyping.Bool[torch.Tensor, "batch seq"],
        expert_weights: jaxtyping.Float[torch.Tensor, "batch seq"],
        output_tensor: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> None:
        """Add weighted expert output to the output tensor."""
        # Extract weights for selected tokens
        selected_weights: jaxtyping.Float[torch.Tensor, "num_tokens"]
        selected_weights = expert_weights[expert_mask]

        # Scale by expert weights
        weighted_output: jaxtyping.Float[torch.Tensor, "num_tokens d_model"]
        weighted_output = expert_output * selected_weights.unsqueeze(-1)

        # Add to output tensor at the right positions
        output_tensor[expert_mask] += weighted_output

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq d_model"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply standard MoE layer.

        The process:
        1. Compute gating scores for all token-expert pairs
        2. Add exploration noise during training
        3. Select top-k experts per token
        4. Normalize expert weights
        5. Route tokens to selected experts
        6. Compute weighted combination of expert outputs
        7. Calculate load balancing loss for training
        """
        scores: jaxtyping.Float[torch.Tensor, "batch seq num_experts"]
        scores = self._compute_gating_scores(x)

        noisy_scores: jaxtyping.Float[torch.Tensor, "batch seq num_experts"]
        noisy_scores = self._add_noise_for_exploration(scores)

        top_k_scores: jaxtyping.Float[torch.Tensor, "batch seq k"]
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq k"]
        top_k_scores, expert_indices = self._select_top_k_experts(
            noisy_scores, self.num_experts_per_token
        )

        expert_weights: jaxtyping.Float[torch.Tensor, "batch seq k"]
        expert_weights = self._normalize_expert_weights(top_k_scores)

        # Initialize output tensor
        output: jaxtyping.Float[torch.Tensor, "batch seq d_model"]
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            expert_mask: jaxtyping.Bool[torch.Tensor, "batch seq"]
            expert_mask = self._find_expert_mask(expert_indices, expert_idx)

            if expert_mask.any():
                expert_output: jaxtyping.Float[torch.Tensor, "num_tokens d_model"]
                expert_output = self._route_tokens_to_expert(x, expert_idx, expert_mask)

                weights_for_expert: jaxtyping.Float[torch.Tensor, "batch seq"]
                weights_for_expert = self._get_expert_weights(
                    expert_weights, expert_indices, expert_idx
                )

                self._combine_expert_outputs(expert_output, expert_mask, weights_for_expert, output)

        if self.training:
            self._aux_loss = self._compute_load_balancing_loss(scores, expert_indices)

        return output


class AuxLossFreeMoE(MoE):
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
