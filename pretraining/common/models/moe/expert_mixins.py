# Third Party
import jaxtyping
import torch
from torch import nn


class ExpertManagementMixin:
    """
    Mixin for routing tokens to experts and combining their outputs.

    Scholarship:
    DeepSeekMoE, 2024, https://arxiv.org/pdf/2401.06066
    Deepseek-V2, 2024, https://arxiv.org/pdf/2405.04434
    Deepseek-V3, 2025, https://arxiv.org/pdf/2412.19437

    Significance:
    Manages the actual expert computation and weighted aggregation.
    Enables fine-grained specialization by routing tokens to multiple experts.
    Efficient batched processing despite irregular routing patterns.

    Init:
    The experts are defined in AuxLossFreeMoE as:
        self.experts = nn.ModuleList([
            SwiGLU(hidden_dim, intermediate_dim, ...)
            for _ in range(num_experts)
        ])

    Step-by-step control flow (_apply_experts_to_tokens):
    1. Initialize output tensor with zeros (same shape as input)
    2. Loop through each of the k expert slots (top-k positions)
    3. Extract expert index and weight for current slot
    4. Loop through all experts to find tokens assigned to each
    5. Gather tokens assigned to current expert into batch
    6. Process batch through expert's SwiGLU network
    7. Scale expert output by routing weight
    8. Add weighted output back to corresponding positions

    Learning process:
    - Expert networks (self.experts: nn.ModuleList of SwiGLU):
      - Each expert learns through standard backpropagation
      - When token prediction is wrong: loss produces gradients
      - Gradients flow back through weighted combination
      - Only experts that processed the token receive gradients
      - Gradient magnitude scaled by routing weight (higher weight = stronger gradient)
      - Result: each expert learns patterns specific to tokens it processes

    - Specialization dynamics:
      - Experts processing similar tokens develop similar representations
      - Experts processing different tokens diverge in their parameters
      - Routing weights modulate learning speed per expert
      - Result: automatic emergence of expert specialization

    - Weighted aggregation:
      - Multiple experts contribute to each token's representation
      - Weights determined by centroid affinity (from routing mixin)
      - Smooth blending allows gradient flow to all selected experts
      - Result: robust representations from ensemble of specialists
    """

    def _apply_experts_to_tokens(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        expert_indices: jaxtyping.Int[torch.Tensor, "batch seq_len num_experts_per_token"],
        expert_weights: jaxtyping.Float[torch.Tensor, "batch seq_len num_experts_per_token"],
        experts: nn.ModuleList,
        num_experts_per_token: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Route tokens to experts and combine outputs."""
        batch_size, seq_len, hidden_dim = x.shape

        # Initialize output
        output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        output = torch.zeros_like(x)

        # Process top-k experts for each token
        for i in range(num_experts_per_token):
            # Get expert assignment for this position
            expert_idx_per_token: jaxtyping.Int[torch.Tensor, "batch seq"]
            expert_idx_per_token = expert_indices[:, :, i]

            expert_weight_per_token: jaxtyping.Float[torch.Tensor, "batch seq"]
            expert_weight_per_token = expert_weights[:, :, i]

            # Process each expert
            for expert_idx in range(len(experts)):
                # Find tokens assigned to this expert
                mask: jaxtyping.Bool[torch.Tensor, "batch seq"]
                mask = expert_idx_per_token == expert_idx

                if mask.any():
                    # Get tokens for this expert
                    expert_input: jaxtyping.Float[torch.Tensor, "num_tokens hidden_dim"]
                    expert_input = x[mask]

                    # Apply expert
                    expert_output: jaxtyping.Float[torch.Tensor, "num_tokens hidden_dim"]
                    expert_output = experts[expert_idx](expert_input)

                    # Scale by gating weight
                    weights: jaxtyping.Float[torch.Tensor, "num_tokens 1"]
                    weights = expert_weight_per_token[mask].unsqueeze(-1)

                    weighted_output: jaxtyping.Float[torch.Tensor, "num_tokens hidden_dim"]
                    weighted_output = expert_output * weights

                    # Add to output
                    output[mask] += weighted_output

        return output


class SharedExpertMixin:
    """
    Mixin for shared experts that process all tokens unconditionally.

    Scholarship:
    Rajbhandari et al., 2022 (prototype concept)
    DeepSeekMoE, 2024, https://arxiv.org/pdf/2401.06066
    Deepseek-V2, 2024, https://arxiv.org/pdf/2405.04434
    Deepseek-V3, 2025, https://arxiv.org/pdf/2412.19437

    Significance:
    Captures common knowledge that all tokens need, reducing redundancy in routed experts.
    Provides guaranteed baseline processing even if routing fails completely.
    Allows routed experts to focus on specialized patterns rather than basics.

    Init:
    The shared expert is defined in AuxLossFreeMoE as:
        self.shared_expert = SwiGLU(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim * n_shared_experts,  # n_shared_experts × capacity
            ...
        )

    Step-by-step control flow (_apply_shared_expert):
    1. Receive input tokens of shape [batch, seq_len, hidden_dim]
    2. Pass ALL tokens through shared expert network
    3. Multiply output by shared_expert_ratio (typically 0.1)
    4. Return scaled output for addition with routed expert outputs

    Learning process:
    - Shared expert network (self.shared_expert: SwiGLU):
      - Processes every single token in every batch
      - When any token is mispredicted: receives gradient signal
      - Gradients averaged across all tokens in batch
      - Learns representations useful for most/all tokens
      - Weight updates reflect common patterns across entire dataset
      - Result: captures general linguistic knowledge and basic transformations

    - Knowledge consolidation:
      - Sees much more data than any single routed expert
      - Learns robust, general-purpose representations
      - Reduces pressure on routed experts to learn basics
      - Result: routed experts can specialize more effectively

    - Scaling factor (shared_expert_ratio):
      - Fixed at 0.1, not learned
      - Prevents shared expert from dominating output
      - Ensures routed experts contribute most of the signal
      - Result: balanced contribution between general and specialized knowledge
    """

    def _apply_shared_expert(
        self,
        x: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"],
        shared_expert: nn.Module,
        shared_expert_ratio: float,
    ) -> jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]:
        """Apply shared expert to all tokens."""
        shared_output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        shared_output = shared_expert(x)

        # Scale by shared expert ratio
        scaled_output: jaxtyping.Float[torch.Tensor, "batch seq_len hidden_dim"]
        scaled_output = shared_output * shared_expert_ratio

        return scaled_output
