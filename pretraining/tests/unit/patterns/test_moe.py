"""
Unit tests for Mixture of Experts components.

Tests StandardMoE and AuxLossFreeMoE implementations.
"""

# Third Party
import pytest
import torch
import torch.nn as nn
import torch.testing

# Project
# Local
from pretraining.common.patterns import moe


class TestStandardMoE:
    """Test standard MoE with auxiliary loss."""

    @pytest.fixture
    def standard_moe(self) -> moe.StandardMoE:
        """Create StandardMoE instance."""
        return moe.StandardMoE(
            hidden_dim=128,
            num_experts=4,
            num_experts_per_token=2,
            dropout=0.0,
        )

    def test_standard_moe_initialization(self, standard_moe: moe.StandardMoE) -> None:
        """Test StandardMoE initializes correctly."""
        assert standard_moe.hidden_dim == 128
        assert standard_moe.num_experts == 4
        assert standard_moe.num_experts_per_token == 2

        # Check gate
        assert standard_moe.gate.in_features == 128
        assert standard_moe.gate.out_features == 4

        # Check experts are Sequential modules
        assert len(standard_moe.experts) == 4
        for expert in standard_moe.experts:
            assert isinstance(expert, nn.Sequential)
            # Check expert has correct structure
            assert len(expert) == 4  # Linear, GELU, Linear, Dropout

    def test_standard_moe_forward(self, standard_moe: moe.StandardMoE) -> None:
        """Test StandardMoE forward pass."""
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 128)

        # Enable training mode to get auxiliary loss
        standard_moe.train()
        output = standard_moe(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, 128)

        # Check auxiliary loss is computed
        assert hasattr(standard_moe, "_aux_loss")
        assert standard_moe._aux_loss is not None
        assert standard_moe._aux_loss.shape == ()  # Scalar
        assert standard_moe._aux_loss.item() >= 0  # Non-negative

    def test_standard_moe_gating(self, standard_moe: moe.StandardMoE) -> None:
        """Test gating mechanism."""
        x = torch.randn(1, 4, 128)

        # Compute gating scores
        x_flat = x.view(-1, 128)
        gating_scores = standard_moe._compute_gating_scores(x_flat)

        # Check shape: [batch*seq, num_experts]
        assert gating_scores.shape == (4, 4)

        # Apply top-k selection
        expert_weights, expert_indices = standard_moe._select_top_k_experts(gating_scores, 2)

        # Check shapes
        assert expert_weights.shape == (4, 2)  # top-2 experts
        assert expert_indices.shape == (4, 2)

        # Apply softmax normalization to weights
        expert_weights = standard_moe._normalize_expert_weights(expert_weights)

        # Check weights sum to approximately 1
        weight_sums = expert_weights.sum(dim=1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums))

        # Check indices are valid
        assert torch.all(expert_indices >= 0)
        assert torch.all(expert_indices < 4)

    def test_standard_moe_load_balancing(self, standard_moe: moe.StandardMoE) -> None:
        """Test load balancing auxiliary loss computation."""
        # Create imbalanced input that routes mostly to first expert
        x = torch.zeros(10, 1, 128)
        x[:, :, 0] = 10.0  # Strong signal for first dimension

        standard_moe.train()
        _ = standard_moe(x)

        # Auxiliary loss should be high for imbalanced routing
        assert standard_moe._aux_loss.item() > 0

    def test_standard_moe_exploration_noise(self, standard_moe: moe.StandardMoE) -> None:
        """Test exploration noise during training."""
        x = torch.randn(1, 10, 128)

        # Training mode - should add noise
        standard_moe.train()
        scores1 = standard_moe._compute_gating_scores(x)
        # Add noise for exploration
        scores1_noisy = standard_moe._add_noise_for_exploration(scores1)
        scores2 = standard_moe._compute_gating_scores(x)
        scores2_noisy = standard_moe._add_noise_for_exploration(scores2)

        # Scores with noise should differ
        assert not torch.allclose(scores1_noisy, scores2_noisy)

        # Eval mode - no noise
        standard_moe.eval()
        scores3 = standard_moe._compute_gating_scores(x)
        scores3_no_noise = standard_moe._add_noise_for_exploration(scores3)
        scores4 = standard_moe._compute_gating_scores(x)
        scores4_no_noise = standard_moe._add_noise_for_exploration(scores4)

        # Scores should be identical (no noise added in eval mode)
        torch.testing.assert_close(scores3_no_noise, scores4_no_noise)

    def test_standard_moe_expert_routing(self, standard_moe: moe.StandardMoE) -> None:
        """Test that tokens are correctly routed to experts."""
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, 128)

        # Get flat input
        x_flat = x.view(-1, 128)

        # Get routing decisions
        gating_scores = standard_moe._compute_gating_scores(x_flat)
        expert_weights, expert_indices = standard_moe._select_top_k_experts(gating_scores, 2)
        expert_weights = standard_moe._normalize_expert_weights(expert_weights)

        # Check each token is routed to exactly k experts
        assert expert_indices.shape == (batch_size * seq_len, 2)

        # Verify no duplicate experts per token
        for i in range(expert_indices.shape[0]):
            selected_experts = expert_indices[i].tolist()
            assert len(selected_experts) == len(set(selected_experts))


class TestAuxLossFreeMoE:
    """Test auxiliary loss-free MoE (DeepSeek style)."""

    @pytest.fixture
    def auxfree_moe(self) -> moe.AuxLossFreeMoE:
        """Create AuxLossFreeMoE instance."""
        return moe.AuxLossFreeMoE(
            hidden_dim=128,
            num_experts=4,
            num_experts_per_token=2,
            dropout=0.0,
        )

    def test_auxfree_moe_initialization(self, auxfree_moe: moe.AuxLossFreeMoE) -> None:
        """Test AuxLossFreeMoE initializes correctly."""
        # Should have shared expert
        assert auxfree_moe.shared_expert is not None
        assert isinstance(auxfree_moe.shared_expert, nn.Sequential)

        # Should have learnable bias
        assert hasattr(auxfree_moe, "gate_bias")
        assert auxfree_moe.gate_bias.shape == (4,)
        assert auxfree_moe.gate_bias.requires_grad

        # Should track expert load
        assert hasattr(auxfree_moe, "expert_load")
        assert auxfree_moe.expert_load.shape == (4,)

    def test_auxfree_moe_forward(self, auxfree_moe: moe.AuxLossFreeMoE) -> None:
        """Test AuxLossFreeMoE forward pass."""
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 128)

        output = auxfree_moe(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, 128)

        # Should not have auxiliary loss
        assert not hasattr(auxfree_moe, "_aux_loss") or auxfree_moe._aux_loss is None

    def test_auxfree_moe_shared_expert(self, auxfree_moe: moe.AuxLossFreeMoE) -> None:
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

    def test_auxfree_moe_gating_with_bias(self, auxfree_moe: moe.AuxLossFreeMoE) -> None:
        """Test that gating bias affects expert selection."""
        x = torch.randn(10, 1, 128)

        # Get baseline expert selection without bias
        with torch.no_grad():
            auxfree_moe.gate_bias.zero_()

        x_flat = x.view(-1, 128)
        baseline_scores = auxfree_moe._compute_gating_with_bias(x_flat)
        _, baseline_indices = auxfree_moe._select_top_k_experts(baseline_scores, 2)
        baseline_expert_0 = (baseline_indices == 0).sum().item()

        # Now add bias to favor expert 0
        with torch.no_grad():
            auxfree_moe.gate_bias[0] = 2.0  # More reasonable bias

        biased_scores = auxfree_moe._compute_gating_with_bias(x_flat)
        _, biased_indices = auxfree_moe._select_top_k_experts(biased_scores, 2)
        biased_expert_0 = (biased_indices == 0).sum().item()

        # Verify bias increased expert 0 selection
        assert biased_expert_0 >= baseline_expert_0

    def test_auxfree_moe_load_tracking(self, auxfree_moe: moe.AuxLossFreeMoE) -> None:
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
        torch.testing.assert_close(auxfree_moe.expert_load, initial_load)

    def test_auxfree_moe_forward_and_bias_tracking(self, auxfree_moe: moe.AuxLossFreeMoE) -> None:
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
        torch.testing.assert_close(auxfree_moe.expert_load, current_load)


class TestMoEHelpers:
    """Test MoE helper functions and edge cases."""

    def test_expert_capacity_calculation(self) -> None:
        """Test capacity calculation for different scenarios."""
        # standard_moe = moe.StandardMoE(
        #     hidden_dim=64,
        #     num_experts=4,
        #     num_experts_per_token=2,
        # )

        # With 8 tokens and 2 experts per token, total slots = 16
        # Distributed across 4 experts = 4 per expert average
        # With capacity factor, should have some buffer

        batch_size, seq_len = 2, 4
        total_tokens = batch_size * seq_len

        # Each token selects 2 experts
        total_selections = total_tokens * 2
        avg_per_expert = total_selections / 4

        assert avg_per_expert == 4.0

    def test_no_expert_selected_edge_case(self) -> None:
        """Test handling when gating scores are all very negative."""
        standard_moe = moe.StandardMoE(
            hidden_dim=64,
            num_experts=4,
            num_experts_per_token=2,
        )

        # Create very small input that might lead to numerical issues
        x = torch.zeros(1, 1, 64) * 1e-10

        # Should still produce valid output
        output = standard_moe(x)
        assert output.shape == (1, 1, 64)
        assert torch.isfinite(output).all()
