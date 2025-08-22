# Third Party
import jaxtyping
import torch
from torch import nn
from torch.nn import functional as F


class DynamicBiasLoadBalancingMixin:
    """
    Mixin for auxiliary-loss-free load balancing through dynamic bias adjustment.

    Scholarship:
    Shazeer et al., 2017, https://openreview.net/forum?id=B1ckMDqlg
        - unbalanced loads lead to collapse
    Fedus et al., 2021, https://arxiv.org/abs/2101.03961
        - traditional aux loss routing
    Lepikhin et al., 2021, https://openreview.net/forum?id=qrwe7XHTmYb
        - traditional aux loss routing
    Wang et al., 2024, https://doi.org/10.48550/arXiv.2408.15664
        - high aux loss will disrupt training
    Deepseek-V3, 2025, https://arxiv.org/pdf/2412.19437

    Significance:
    Achieves load balance without disrupting training with large auxiliary losses.
    Bias terms act like traffic signals, redirecting tokens away from busy experts.
    Enables better model performance by avoiding the auxiliary loss trade-off.

    Init:
    The components are defined in AuxLossFreeMoE as:
        self.gate_bias = nn.Parameter(torch.zeros(num_experts))  # Learnable routing bias
        self.expert_load = torch.zeros(num_experts)  # Buffer for tracking cumulative load

    Step-by-step control flow (_compute_expert_load):
    1. Receive expert indices showing which experts were selected for each token
    2. Convert indices to one-hot encoding
    3. Sum across batch and sequence dimensions
    4. Return total count of tokens assigned to each expert

    Step-by-step control flow (_update_load_balancing_bias):
    1. Count how many tokens went to each expert in current batch
    2. Update running average of expert load (blend 90% old, 10% new)
    3. Calculate mean load across all experts
    4. Find load imbalance for each expert (actual - mean)
    5. Adjust bias: decrease for overloaded, increase for underloaded

    Learning process:
    - This mixin contains learnable parameters but they don't learn through gradients.

    - Gate bias (self.gate_bias: nn.Parameter):
      - Does NOT receive gradients from the loss (wrapped in torch.no_grad())
      - Updates based on load statistics, not backpropagation
      - Overloaded expert: bias decreases by (update_speed * excess_load)
      - Underloaded expert: bias increases by (update_speed * deficit_load)
      - Result: bias values self-adjust to equalize expert usage

    - Load tracking (self.expert_load: Buffer):
      - Maintains exponential moving average of expert usage
      - Smooths out batch-to-batch variations
      - Provides stable signal for bias adjustments
      - Result: system reaches equilibrium where all experts process similar loads

    - System dynamics:
      - No gradient interference with model learning
      - Pure algorithmic load balancing, not learned
      - Bias only affects routing selection, not expert weights
      - Result: achieves balance without sacrificing model quality
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
