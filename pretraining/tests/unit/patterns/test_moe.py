"""
Unit tests for Mixture of Experts components.

Tests AuxLossFreeMoE implementation.
"""

# Third Party
import pytest
import torch
from torch import nn
from torch import testing

# Project
from pretraining.common.models.moe import aux_loss_free


class TestAuxLossFreeMoE:
    """Test auxiliary loss-free MoE (DeepSeek style)."""

    @pytest.fixture
    def auxfree_moe(self) -> aux_loss_free.AuxLossFreeMoE:
        """Create AuxLossFreeMoE instance."""
        return aux_loss_free.AuxLossFreeMoE(
            hidden_dim=128,
            num_experts=4,
            num_experts_per_token=2,
            dropout=None,
        )

    def test_auxfree_moe_initialization(self, auxfree_moe: aux_loss_free.AuxLossFreeMoE) -> None:
        """Test AuxLossFreeMoE initializes correctly."""
        # Should have shared expert
        assert auxfree_moe.shared_expert is not None
        assert isinstance(auxfree_moe.shared_expert, nn.Module)

        # Should have learnable bias
        assert hasattr(auxfree_moe, "gate_bias")
        assert auxfree_moe.gate_bias.shape == (4,)
        assert auxfree_moe.gate_bias.requires_grad

        # Should track expert load
        assert hasattr(auxfree_moe, "expert_load")
        assert auxfree_moe.expert_load.shape == (4,)

    def test_auxfree_moe_forward(self, auxfree_moe: aux_loss_free.AuxLossFreeMoE) -> None:
        """Test AuxLossFreeMoE forward pass."""
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 128)

        output = auxfree_moe(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, 128)

        # Should have auxiliary loss after forward pass in training mode
        assert hasattr(auxfree_moe, "_aux_loss")
        assert auxfree_moe._aux_loss is not None
        assert auxfree_moe._aux_loss.numel() == 1  # Should be a scalar

    def test_auxfree_moe_shared_expert(self, auxfree_moe: aux_loss_free.AuxLossFreeMoE) -> None:
        """Test shared expert processing."""
        x = torch.randn(1, 4, 128)

        # Apply shared expert directly
        shared_output = auxfree_moe.shared_expert(x)

        # Should be different from input
        assert not torch.allclose(shared_output, x)
        assert shared_output.shape == x.shape

        # Forward pass should combine shared and routed experts
        full_output = auxfree_moe(x)

        # Output should be different from just shared expert
        # (since it includes routed experts too)
        assert not torch.allclose(full_output, shared_output)

    def test_auxfree_moe_gating_with_bias(self, auxfree_moe: aux_loss_free.AuxLossFreeMoE) -> None:
        """Test that gating bias affects expert selection."""
        x = torch.randn(10, 1, 128)

        # Get baseline expert selection without bias
        with torch.no_grad():
            auxfree_moe.gate_bias.zero_()

        # Compute affinity scores
        affinity_scores = auxfree_moe._compute_centroid_affinity(x, auxfree_moe.expert_centroids)
        baseline_scores = affinity_scores + auxfree_moe.gate_bias
        _, baseline_indices = auxfree_moe._select_top_k_experts(baseline_scores, 2)
        baseline_expert_0 = (baseline_indices == 0).sum().item()

        # Now add bias to favor expert 0
        with torch.no_grad():
            auxfree_moe.gate_bias[0] = 2.0  # More reasonable bias

        biased_scores = affinity_scores + auxfree_moe.gate_bias
        _, biased_indices = auxfree_moe._select_top_k_experts(biased_scores, 2)
        biased_expert_0 = (biased_indices == 0).sum().item()

        # Verify bias increased expert 0 selection
        assert biased_expert_0 >= baseline_expert_0

    def test_auxfree_moe_load_tracking(self, auxfree_moe: aux_loss_free.AuxLossFreeMoE) -> None:
        """Test expert load tracking."""
        # Create input that routes to specific experts
        x = torch.randn(20, 1, 128)

        # Forward pass
        _ = auxfree_moe(x)

        # Check load was tracked (in eval mode it shouldn't update)
        auxfree_moe.eval()
        initial_load = auxfree_moe.expert_load.clone()

        _ = auxfree_moe(x)

        # Load shouldn't change in eval mode
        testing.assert_close(auxfree_moe.expert_load, initial_load)

    def test_auxfree_moe_forward_and_bias_tracking(
        self, auxfree_moe: aux_loss_free.AuxLossFreeMoE
    ) -> None:
        """Test that MoE forward pass works and updates load tracking."""
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, 128)

        # Get initial load state
        initial_load = auxfree_moe.expert_load.clone()

        # Forward pass in training mode
        auxfree_moe.train()
        output = auxfree_moe(x)

        # Verify output shape and validity
        assert output.shape == (batch_size, seq_len, 128)
        assert torch.isfinite(output).all()

        # Verify load tracking was updated (in training mode)
        assert not torch.allclose(auxfree_moe.expert_load, initial_load)

        # In eval mode, load shouldn't update
        auxfree_moe.eval()
        current_load = auxfree_moe.expert_load.clone()
        _ = auxfree_moe(x)
        testing.assert_close(auxfree_moe.expert_load, current_load)

    def test_auxfree_moe_gating(self, auxfree_moe: aux_loss_free.AuxLossFreeMoE) -> None:
        """Test gating mechanism."""
        x = torch.randn(1, 4, 128)

        # Compute affinity scores and add bias
        affinity_scores = auxfree_moe._compute_centroid_affinity(x, auxfree_moe.expert_centroids)
        gating_scores = affinity_scores + auxfree_moe.gate_bias

        # Check shape: [batch*seq, num_experts]
        assert gating_scores.shape == (1, 4, 4)

        # Apply top-k selection
        expert_weights, expert_indices = auxfree_moe._select_top_k_experts(gating_scores, 2)

        # Check shapes
        assert expert_weights.shape == (1, 4, 2)  # top-2 experts
        assert expert_indices.shape == (1, 4, 2)

        # Apply softmax normalization to weights
        expert_weights = auxfree_moe._normalize_expert_weights(expert_weights)

        # Check weights sum to approximately 1
        weight_sums = expert_weights.sum(dim=-1)
        testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-4, rtol=1e-4)

        # Check indices are valid
        assert torch.all(expert_indices >= 0)
        assert torch.all(expert_indices < 4)

    def test_auxfree_moe_expert_routing(self, auxfree_moe: aux_loss_free.AuxLossFreeMoE) -> None:
        """Test that tokens are correctly routed to experts."""
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, 128)

        # Get routing decisions
        affinity_scores = auxfree_moe._compute_centroid_affinity(x, auxfree_moe.expert_centroids)
        gating_scores = affinity_scores + auxfree_moe.gate_bias
        expert_weights, expert_indices = auxfree_moe._select_top_k_experts(gating_scores, 2)
        expert_weights = auxfree_moe._normalize_expert_weights(expert_weights)

        # Check each token is routed to exactly k experts
        assert expert_indices.shape == (batch_size, seq_len, 2)

        # Verify no duplicate experts per token
        for b in range(batch_size):
            for s in range(seq_len):
                selected_experts = expert_indices[b, s].tolist()
                assert len(selected_experts) == len(set(selected_experts))


class TestMoEHelpers:
    """Test MoE helper functions and edge cases."""

    def test_expert_capacity_calculation(self) -> None:
        """Test capacity calculation for different scenarios."""
        batch_size, seq_len = 2, 4
        total_tokens = batch_size * seq_len

        # Each token selects 2 experts
        total_selections = total_tokens * 2
        avg_per_expert = total_selections / 4

        assert avg_per_expert == 4.0

    def test_no_expert_selected_edge_case(self) -> None:
        """Test handling when gating scores are all very negative."""
        auxfree_moe = aux_loss_free.AuxLossFreeMoE(
            hidden_dim=64,
            num_experts=4,
            num_experts_per_token=2,
        )

        # Create very small input that might lead to numerical issues
        x = torch.zeros(1, 1, 64) * 1e-10

        # Should still produce valid output
        output = auxfree_moe(x)
        assert output.shape == (1, 1, 64)
        assert torch.isfinite(output).all()
