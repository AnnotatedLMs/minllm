# Third Party
import jaxtyping
import torch
from torch.nn import functional as F


class AuxiliaryLossMixin:
    """
    Mixin for complementary sequence-wise balance loss computation.

    Scholarship:
    Deepseek-V3, 2025, https://arxiv.org/pdf/2412.19437

    Significance:
    Provides safety net against extreme within-sequence imbalance.
    Uses tiny alpha (0.001) to avoid disrupting main training dynamics.
    Acts as backup when bias-based balancing isn't enough.

    Init:
    This mixin has no initialization. It's a pure computation module.
    The alpha factor is defined in AuxLossFreeMoE as:
        self.aux_loss_alpha = 0.001  # "extremely small" per paper

    Step-by-step control flow (_compute_auxiliary_loss):
    1. Convert expert indices to one-hot encoding
    2. Count how many times each expert was selected (indicator sum)
    3. Normalize counts to get fraction f_i for each expert
    4. Compute normalized affinity scores s_prime for each expert
    5. Average s_prime across sequence to get probability P_i
    6. Compute balance loss as sum(f_i * P_i)
    7. Average across batch for final loss value

    Learning process:
    - This mixin contains no learnable parameters.

    - Loss computation:
      - Measures discrepancy between actual usage (f_i) and intended usage (P_i)
      - When balanced: f_i ≈ P_i for all experts, loss is minimal
      - When imbalanced: some experts have high f_i but low P_i (or vice versa)
      - Loss increases quadratically with imbalance severity

    - Gradient effects (through aux_loss_alpha * loss):
      - Extremely weak signal due to alpha = 0.001
      - Slightly adjusts expert centroids to discourage extreme imbalance
      - Does not interfere with primary learning objectives
      - Result: prevents pathological routing patterns without harming performance

    - Interaction with bias-based balancing:
      - Bias adjustment handles most load balancing (no gradients)
      - Auxiliary loss provides gradient-based correction for edge cases
      - Together they ensure robust load distribution
      - Result: system maintains balance even in challenging scenarios
    """

    def _compute_auxiliary_loss(
        self,
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq_len num_experts_per_token"],
        affinity_scores: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"],
        num_experts: int,
        num_experts_per_token: int,
    ) -> torch.Tensor:
        """Compute auxiliary loss for logging/metrics."""
        batch_size, seq_len, _ = expert_indices.shape

        # Compute f_i: fraction of tokens assigned to each expert
        one_hot: jaxtyping.Float[torch.Tensor, "batch seq num_experts_per_token num_experts"]
        one_hot = F.one_hot(expert_indices, num_experts).float()

        indicator: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts"]
        indicator = one_hot.sum(dim=2)

        f_i: jaxtyping.Float[torch.Tensor, "batch num_experts"]
        f_i = indicator.sum(dim=1) * (num_experts / (num_experts_per_token * seq_len))

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
