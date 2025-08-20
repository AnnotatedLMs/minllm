# Third Party
import jaxtyping
import torch
from torch import nn
from torch.nn import functional as F


class DynamicBiasLoadBalancingMixin:
    """
    Mixin for dynamic bias adjustment to balance expert load.
    https://arxiv.org/pdf/2412.19437

    Based on evidence from:
    Shazeer et al., 2017 - https://openreview.net/forum?id=B1ckMDqlg
        - unbalanced loads lead to collapse
    Fedus et al., 2021, Lepikhin et al., 2021 - https://arxiv.org/abs/2101.03961 , https://openreview.net/forum?id=qrwe7XHTmYb
        - traditional aux loss routing
    Wang et al., 2024 - https://doi.org/10.48550/arXiv.2408.15664.
        - high aux loss will disrupt training

    Significance:
    Load balancing without using auxiliary loss as the primary mechanism.
    Bias terms automatically adjust to steer tokens away from overloaded experts.

    Init:
    The bias terms are defined in AuxLossFreeMoE as:
        self.gate_bias = nn.Parameter(torch.zeros(num_experts))
        self.expert_load = torch.zeros(num_experts)  # Buffer for tracking load

    Routing approach:
    - Each expert has a learnable bias term added to routing scores
    - Overloaded experts get negative bias (become less attractive)
    - Underloaded experts get positive bias (become more attractive)
    - Bias affects routing decisions but NOT the final gating weights

    Step-by-step control flow:
    1. Count how many tokens each expert received in current batch
    2. Update running average of expert load (exponential moving average with factor 0.1)
    3. Compare each expert's load to mean load across all experts
    4. Decrease bias for overloaded experts by (bias_update_speed * load_imbalance)
    5. Increase bias for underloaded experts by (bias_update_speed * load_imbalance)

    Learning process:
    - No gradients flow through bias adjustments (uses torch.no_grad())
    - Bias values evolve based on load statistics, not loss gradients
    - System finds equilibrium where all experts get similar load

    Used by: DeepSeek-V3's AuxLossFreeMoE
    """

    def _compute_expert_load(
        self,
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq_len num_experts_per_token"],
        num_experts: int,
    ) -> jaxtyping.Float[torch.Tensor, "num_experts"]:
        """Compute how many tokens are assigned to each expert."""
        # One-hot encode expert assignments
        one_hot: jaxtyping.Float[torch.Tensor, "batch seq num_experts_per_token num_experts"]
        one_hot = F.one_hot(expert_indices, num_experts).float()

        # Sum across all dimensions to get total load per expert
        expert_load: jaxtyping.Float[torch.Tensor, "num_experts"]
        expert_load = one_hot.sum(dim=[0, 1, 2])

        return expert_load

    def _update_load_balancing_bias(
        self,
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq_len num_experts_per_token"],
        expert_load_tracker: torch.Tensor,
        gate_bias: nn.Parameter,
        num_experts: int,
        bias_update_speed: float = 0.001,
    ) -> None:
        """Update bias terms based on expert load (no auxiliary loss)."""
        # Count how many tokens go to each expert
        current_load: jaxtyping.Float[torch.Tensor, "num_experts"]
        current_load = self._compute_expert_load(expert_indices, num_experts)

        with torch.no_grad():
            # Exponential moving average of load
            expert_load_tracker.lerp_(current_load, 0.1)

            # Compute load imbalance
            mean_load: torch.Tensor = expert_load_tracker.mean()
            load_imbalance: jaxtyping.Float[torch.Tensor, "num_experts"]
            load_imbalance = expert_load_tracker - mean_load

            # Adjust bias - negative for overloaded experts
            gate_bias.data -= bias_update_speed * load_imbalance
