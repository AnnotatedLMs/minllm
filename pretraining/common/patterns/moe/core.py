# Standard Library
import typing

# Third Party
import jaxtyping
import torch
import torch.nn.functional as F

# Project
from pretraining.common.base import moe


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
        dropout: typing.Optional[float] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout

        # For auxiliary loss tracking
        self._aux_loss = None

    def _compute_gating_scores(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]:
        """
        Compute raw gating scores for each token-expert pair.

        Standard implementation uses linear projection through the learned gate.
        Each token gets a score for every expert.
        """
        scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        scores = self.gate(x)
        return scores

    def _add_noise_for_exploration(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"],
        noise_scale: float = 0.01,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]:
        """
        Add noise during training for exploration.

        Standard implementation adds Gaussian noise.
        """
        if self.training:
            noise: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
            noise = torch.randn_like(scores) * noise_scale

            noisy_scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
            noisy_scores = scores + noise

            return noisy_scores
        return scores

    def _select_top_k_experts(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"],
        k: int,
    ) -> typing.Tuple[
        jaxtyping.Float[torch.Tensor, "batch seq_len k"],
        jaxtyping.Int[torch.Tensor, "batch seq_len k"],
    ]:
        """
        Select top-k experts for each token.

        Standard implementation uses torch.topk.
        """
        top_k_scores: jaxtyping.Float[torch.Tensor, "batch seq_len k"]
        top_k_indices: jaxtyping.Int[torch.Tensor, "batch seq_len k"]
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=-1)

        return top_k_scores, top_k_indices

    def _normalize_expert_weights(
        self,
        scores: jaxtyping.Float[torch.Tensor, "batch seq_len k"],
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len k"]:
        """
        Normalize expert weights to sum to 1.

        Standard implementation uses softmax.
        """
        weights: jaxtyping.Float[torch.Tensor, "batch seq_len k"]
        weights = F.softmax(scores, dim=-1)
        return weights

    def get_auxiliary_loss(self) -> typing.Optional[torch.Tensor]:
        """Get auxiliary loss from last forward pass."""
        return self._aux_loss
