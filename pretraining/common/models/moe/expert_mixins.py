# Third Party
import jaxtyping
import torch
from torch import nn


class ExpertManagementMixin:
    """
    Mixin for creating and managing expert networks.

    Significance:
    Routes tokens to selected experts and combines their outputs.
    Handles the actual computation of expert outputs and weighted aggregation.

    Init:
    The experts are defined in AuxLossFreeMoE as:
        self.experts = nn.ModuleList([SwiGLU(...) for _ in range(num_experts)])

    Routing approach:
    - Each token is processed by its selected experts
    - Expert outputs are weighted by normalized affinity scores
    - Final output is weighted sum of expert outputs

    Step-by-step control flow:
    1. Initialize output tensor with zeros
    2. For each of the top-k expert positions:
       a. Get which expert was selected for each token
       b. Get the weight for that expert
    3. For each expert in the model:
       a. Find all tokens assigned to this expert
       b. Process those tokens through the expert
       c. Multiply by weights and add to output
    4. Return combined output from all experts

    Learning process:
    - Experts learn through standard backpropagation
    - Each expert specializes based on tokens it processes
    - Weights come from routing decisions (centroid affinity)

    Used by: DeepSeek-V3's AuxLossFreeMoE
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
    Mixin for shared expert that processes all tokens.

    Significance:
    Guarantees all tokens get some expert processing, even if routing fails.
    Acts as a baseline transformation that all tokens receive.

    Init:
    The shared expert is defined in AuxLossFreeMoE as:
        self.shared_expert = SwiGLU(intermediate_dim=intermediate_dim * n_shared_experts)
    Note: shared expert has n_shared_experts times the capacity of a single routed expert

    Routing approach:
    - No routing - processes ALL tokens
    - Output scaled by shared_expert_ratio (typically 0.1)
    - Combined additively with routed expert outputs

    Step-by-step control flow:
    1. Pass all tokens through shared expert
    2. Scale output by shared_expert_ratio
    3. Return scaled output (will be added to routed expert outputs)

    Learning process:
    - Learns general patterns that apply to all tokens
    - Provides stable gradient signal for all inputs
    - Acts as regularization against routing failures

    Used by: DeepSeek-V3's AuxLossFreeMoE
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
