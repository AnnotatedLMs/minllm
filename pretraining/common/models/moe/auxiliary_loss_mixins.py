# Third Party
import jaxtyping
import torch
from torch.nn import functional as F


class AuxiliaryLossMixin:
    """
    Mixin for computing auxiliary loss for MoE load balancing.

    Derived from:
    Deepseek-V3, https://arxiv.org/pdf/2412.19437 Section 2.1.2

    Significance:
    Complementary sequence-wise balance loss to prevent extreme imbalance within single sequences.
    Uses "extremely small" alpha factor (0.001) to minimally affect training.

    Init:
    No initialization needed - pure computation mixin.
    The alpha factor is defined in AuxLossFreeMoE as:
        self.aux_loss_alpha = 0.001

    Routing approach:
    - Computes imbalance between how many tokens go to each expert (f_i)
    - And the average probability assigned to each expert (P_i)
    - Loss encourages these to be similar (balanced load)

    Step-by-step control flow:
    1. Count actual token assignments to each expert (f_i)
    2. Compute average probability each expert received (P_i)
    3. Multiply f_i * P_i and sum across experts
    4. Average across batch to get final loss value
    5. Multiply by aux_loss_alpha (0.001) before adding to main loss

    Learning process:
    - Gradients from this loss are minimal due to tiny alpha
    - Serves as safety mechanism for extreme imbalance cases
    - Primary load balancing comes from bias adjustment, not this loss

    Used by: DeepSeek-V3's AuxLossFreeMoE
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
