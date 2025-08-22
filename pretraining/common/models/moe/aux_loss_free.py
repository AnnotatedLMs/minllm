# Standard Library
import typing

# Third Party
import jaxtyping
import torch
from torch import nn

# Project
from pretraining.common.models.ffn import swiglu
from pretraining.common.models.moe import auxiliary_loss_mixins
from pretraining.common.models.moe import expert_mixins
from pretraining.common.models.moe import load_balancing_mixins
from pretraining.common.models.moe import routing_mixins


class AuxLossFreeMoE(
    nn.Module,
    routing_mixins.CentroidRoutingMixin,
    load_balancing_mixins.DynamicBiasLoadBalancingMixin,
    auxiliary_loss_mixins.AuxiliaryLossMixin,
    expert_mixins.ExpertManagementMixin,
    expert_mixins.SharedExpertMixin,
):
    """
    Auxiliary-Loss-Free Mixture of Experts - Load balancing through dynamic bias adjustment.
    Deepseek-V3, 2025, https://arxiv.org/pdf/2412.19437 Section 2.1.2

    Step-by-step control flow (how mixins work together):
    1. SharedExpertMixin: Process all tokens through shared expert (baseline)
    2. CentroidRoutingMixin: Compute affinity scores between tokens and expert centroids
    3. DynamicBiasLoadBalancingMixin: Add routing bias to balance load
    4. CentroidRoutingMixin: Select top-k experts based on biased scores
    5. CentroidRoutingMixin: Extract clean affinity scores for selected experts
    6. CentroidRoutingMixin: Normalize scores to get gating weights
    7. ExpertManagementMixin: Route tokens to experts, apply weights, combine
    8. DynamicBiasLoadBalancingMixin: Update bias based on load statistics
    9. AuxiliaryLossMixin: Compute sequence-wise balance loss (alpha=0.001)

    Learning process (how each mixin affects training):
    - SharedExpertMixin: Shared expert learns general patterns via backprop
    - CentroidRoutingMixin: Expert centroids learn token affinities via backprop
    - DynamicBiasLoadBalancingMixin: NO learning - algorithmic bias adjustment
    - ExpertManagementMixin: Routed experts learn specialized patterns via backprop
    - AuxiliaryLossMixin: Minimal gradient signal for extreme imbalance cases

    Key implementation detail:
    - Bias affects routing selection but NOT gating weights (Eq. 16 in paper)
    - Auxiliary loss uses "extremely small" alpha to avoid disrupting training
    - Load tracking uses exponential moving average for stability
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        intermediate_dim: typing.Optional[int] = None,
        n_shared_experts: int = 2,
        shared_expert_ratio: float = 0.1,
        activation: str = "silu",
        dropout: typing.Optional[float] = None,
        aux_loss_alpha: float = 0.001,  # "extremely small" value per DeepSeek-V3 paper
        bias_update_speed: float = 0.001,  # γ in the paper, how fast to adjust routing bias
    ):
        super().__init__()

        if intermediate_dim is None:
            intermediate_dim = 4 * hidden_dim

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.n_shared_experts = n_shared_experts
        self.shared_expert_ratio = shared_expert_ratio
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.aux_loss_alpha = aux_loss_alpha
        self.bias_update_speed = bias_update_speed

        # Expert centroids for affinity computation
        self.expert_centroids = nn.Parameter(torch.randn(num_experts, hidden_dim))

        # Bias terms for load balancing, key component of DS3s auxuliary-loss-free load balancing
        # https://arxiv.org/pdf/2412.19437 Section: 2.1.2.
        self.gate_bias = nn.Parameter(torch.zeros(num_experts))

        # Shared experts that always process all tokens
        self.shared_expert = swiglu.SwiGLU(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim * n_shared_experts,
            activation=activation,
            dropout=dropout,
            bias=False,
        )

        # Routed experts using SwiGLU
        self.experts = nn.ModuleList(
            [
                swiglu.SwiGLU(
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    activation=activation,
                    dropout=dropout,
                    bias=False,
                )
                for _ in range(num_experts)
            ]
        )

        # Buffer: tracks cumulative expert load for dynamic bias adjustments
        self.register_buffer("expert_load", torch.zeros(num_experts))

        # For auxiliary loss computation (metrics only)
        self._aux_loss = None

    def get_auxiliary_loss(self) -> typing.Optional[torch.Tensor]:
        """Get auxiliary loss for metrics (not used in training)."""
        return self._aux_loss

    def forward(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Apply auxiliary-loss-free MoE to input tokens."""

        shared_output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        shared_output = self._apply_shared_expert(x, self.shared_expert, self.shared_expert_ratio)

        affinity_scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        affinity_scores = self._compute_centroid_affinity(x, self.expert_centroids)

        scores_with_bias: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        scores_with_bias = self._apply_routing_bias(affinity_scores, self.gate_bias)

        # Select top-k experts based on biased scores
        top_k_scores, top_k_indices = self._select_top_k_experts(
            scores_with_bias, self.num_experts_per_token
        )

        # Extract affinity scores for selected experts (without bias)
        batch_size, seq_len, _ = x.shape
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).unsqueeze(2)
        seq_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(2)

        top_k_affinity: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts_per_token"]
        top_k_affinity = affinity_scores[batch_indices, seq_indices, top_k_indices]

        expert_weights: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts_per_token"]
        expert_weights = self._normalize_expert_weights(top_k_affinity)

        routed_output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        routed_output = self._apply_experts_to_tokens(
            x, top_k_indices, expert_weights, self.experts, self.num_experts_per_token
        )

        if self.training:
            self._update_load_balancing_bias(
                top_k_indices,
                self.expert_load,
                self.gate_bias,
                self.num_experts,
                self.bias_update_speed,
            )

            raw_aux_loss = self._compute_auxiliary_loss(
                top_k_indices,
                affinity_scores,
                self.num_experts,
                self.num_experts_per_token,
            )

            # Apply the extremely small alpha factor as per paper
            # https://arxiv.org/pdf/2412.19437 Section: 2.1.2.
            self._aux_loss = self.aux_loss_alpha * raw_aux_loss

        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = shared_output + routed_output

        return output
