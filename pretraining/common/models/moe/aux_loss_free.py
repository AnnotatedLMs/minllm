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
    DeepSeek-V3 MoE implementation - Auxiliary Loss Free Mixture of Experts.
    https://arxiv.org/pdf/2412.19437

    Derived from:
    - DeepSeekMOE, 2024, https://arxiv.org/pdf/2401.06066,
    - DeepseekV2, 2024, https://arxiv.org/pdf/2405.04434, Section 2.2

    Step-by-step control flow (how mixins work together):
    1. SharedExpertMixin: ALL tokens go through shared expert first (baseline processing)
    2. CentroidRoutingMixin: Compute how similar each token is to each expert's specialty
    3. DynamicBiasLoadBalancingMixin: Add bias to steer tokens away from busy experts
    4. CentroidRoutingMixin: Pick top-k experts based on biased scores
    5. CentroidRoutingMixin: Get clean weights from original scores (no bias)
    6. ExpertManagementMixin: Send tokens to selected experts, combine outputs
    7. DynamicBiasLoadBalancingMixin: Update bias values for next batch
    8. AuxiliaryLossMixin: Compute tiny loss term as safety check

    Learning process (how each mixin affects training):
    - SharedExpertMixin: Learns via normal backprop, sees all tokens
    - CentroidRoutingMixin: Expert centroids learn to attract suitable tokens (backprop)
    - DynamicBiasLoadBalancingMixin: NO learning - just adjusts routing bias
    - ExpertManagementMixin: Each expert learns from its assigned tokens (backprop)
    - AuxiliaryLossMixin: Tiny gradient signal (alpha=0.001) for extreme cases

    The "auxiliary-loss-free" name means load balancing happens primarily through bias
    adjustment (no gradients) rather than through loss terms (has gradients).
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
