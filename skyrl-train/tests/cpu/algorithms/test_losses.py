"""
Tests for policy loss functions.

uv run --isolated --extra dev -- pytest tests/cpu/algorithms/test_losses.py
"""

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    masked_mean,
    TrieNode,
    build_prefix_trie,
    compute_localized_stats,
    compute_branched_grpo_advantage,
)


# Adapted a good test from NeMO-RL
def test_policy_loss_dual_clip():
    """Tests dual clipping in PolicyLoss function."""

    device = "cpu"

    # Create test data with a mix of advantages: positive, slightly negative, strongly negative
    advantages = torch.tensor([[1.0, -1.0, -4.0]], device=device)

    # Set up logprobs to test different probability ratios
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]], device=device)
    log_probs = torch.tensor([[-1.69315, -1.0, -0.69741]], device=device)  # approx log(0.5)-1, log(1)-1, log(10)-3

    # Create config for dual clipping
    config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "dual_clip",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    # Create loss function with dual clipping
    loss_fn = PolicyLossRegistry.get("dual_clip")

    # Calculate expected values
    ratio = torch.exp(log_probs - old_log_probs)  # approx [0.5, 1.0, 10.0]
    assert torch.allclose(ratio, torch.tensor([[0.5, 1.0, 10.0]], device=device), rtol=1e-3)

    # Standard PPO clipping
    loss1 = -ratio * advantages  # [0.5, -1.0, -40.0]
    loss2 = -ratio.clamp(1 - 0.2, 1 + 0.2) * advantages  # [0.8, -1.0, -4.8]
    max_loss = torch.maximum(loss1, loss2)  # [0.5, -1.0, -40.0]

    # Dual clipping
    loss3 = -advantages * 3.0  # [-3.0, 3.0, 12.0]
    min_loss = torch.min(loss3, max_loss)  # [-3.0, 1.0, 12.0]

    # For negative advantages, use dual clipped loss
    final_loss = torch.where(advantages < 0, min_loss, max_loss)  # [-0.5, 1.0, 12.0]
    assert torch.allclose(final_loss, torch.tensor([[-0.5, 1.0, 12.0]], device=device), rtol=1e-3)
    expected_loss = final_loss.mean()  # -(-12.5/3) = 4.1667

    # Calculate actual loss
    actual_loss, _ = loss_fn(log_probs=log_probs, old_log_probs=old_log_probs, advantages=advantages, config=config)

    # Verify results
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-8)
    # close to hand calculated value
    assert actual_loss.item() == pytest.approx(4.1667, abs=1e-4)


def test_policy_loss_cispo():
    """Tests CISPO in PolicyLoss function."""

    device = "cpu"

    # Create test data with a mix of advantages: positive, slightly negative, strongly negative
    advantages = torch.tensor([[1.0, -1.0, -4.0]], device=device)

    # Set up logprobs to test different probability ratios
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]], device=device)
    log_probs = torch.tensor([[-1.69315, -1.0, -0.69741]], device=device)  # approx log(0.5)-1, log(1)-1, log(10)-3

    # Create config for cispo
    config = DictConfig(
        {
            "cispo": {"cispo_eps_clip_low": 0.2, "cispo_eps_clip_high": 0.2},
            "policy_loss_type": "cispo",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    # Create loss function with cispo
    loss_fn = PolicyLossRegistry.get("cispo")

    # Calculate expected values
    ratio = torch.exp(log_probs - old_log_probs)  # approx [0.5, 1.0, 10.0]
    assert torch.allclose(ratio, torch.tensor([[0.5, 1.0, 10.0]], device=device), rtol=1e-3)

    # Hand-calculation for expected loss:
    # ratio = [0.5, 1.0, 10.0]
    # clamped_ratio = ratio.clamp(0.8, 1.2) = [0.8, 1.0, 1.2]
    # advantages = [1.0, -1.0, -4.0]
    # log_probs = [-1.69315, -1.0, -0.69741]
    # loss_per_token = -advantages * clamped_ratio * log_probs
    # loss_per_token[0] = -(1.0 * 0.8 * -1.69315) = 1.35452
    # loss_per_token[1] = -(-1.0 * 1.0 * -1.0) = -1.0
    # loss_per_token[2] = -(-4.0 * 1.2 * -0.69741) = -3.347568
    # mean(loss) = (1.35452 - 1.0 - 3.347568) / 3 = -0.99768266666
    loss = -ratio.clamp(1 - 0.2, 1 + 0.2) * advantages * log_probs
    expected_loss = loss.mean()

    # Calculate actual loss
    actual_loss, _ = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=config,
    )

    # Verify results
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-8)
    # close to hand calculated value
    assert actual_loss.item() == pytest.approx(-0.99768266666, abs=1e-4)


def test_policy_loss_reduction_modes():
    """Tests different loss_reduction modes in PolicyLoss function.

    Note: token_mean and sequence_mean give the same result when all sequences
    have the same length and no mask is applied, but differ when masking creates
    different effective sequence lengths.
    """

    device = "cpu"

    clip_eps_low = 0.2
    clip_eps_high = 0.2

    advantages = torch.tensor(
        [
            [2.0, 2.0, 2.0],  # sequence 1: consistently higher advantages
            [1.0, 1.0, 1.0],  # sequence 2: consistently lower advantages
        ],
        device=device,
    )

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], device=device)

    log_probs = torch.tensor(
        [[-1.5, -0.5, -1.2], [-0.8, -1.3, -0.9]],  # ratios ≈ [[0.61, 1.65, 0.83],[1.22, 0.74, 1.11]]
        device=device,
    )

    # Create masks to test sequences with different numbers of valid tokens
    loss_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], device=device)

    # Create configs for different reduction modes
    config_token = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    config_seq = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "sequence_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    # Get loss function
    loss_fn = PolicyLossRegistry.get("regular")

    # Test token_mean without mask
    loss_token_no_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_token)

    # Test token_mean with mask
    loss_token_with_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_token, loss_mask)

    # Test sequence_mean without mask
    loss_seq_no_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq)

    # Test sequence_mean with mask
    loss_seq_with_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq, loss_mask)

    # Manual calculations to verify (using default PolicyLoss parameters)
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages  # clip_eps_low=0.2, clip_eps_high=0.2
    loss_per_token = -torch.min(surr1, surr2)

    # Expected token_mean without mask: mean of all tokens
    expected_token_no_mask = loss_per_token.mean()

    # Expected token_mean with mask: masked mean of all tokens
    expected_token_with_mask = (loss_per_token * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    # Expected sequence_mean without mask: mean of sequence means
    expected_seq_no_mask = loss_per_token.mean(dim=1).mean()

    # Expected sequence_mean with mask: mean of masked sequence means
    seq_means_masked = (loss_per_token * loss_mask).sum(dim=1) / (loss_mask.sum(dim=1) + 1e-8)
    expected_seq_with_mask = seq_means_masked.mean()

    # Verify results
    torch.testing.assert_close(loss_token_no_mask, expected_token_no_mask, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_token_with_mask, expected_token_with_mask, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_seq_no_mask, expected_seq_no_mask, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_seq_with_mask, expected_seq_with_mask, rtol=1e-5, atol=1e-8)

    # Verify that the two reduction modes give the same results when sequences have equal length and no mask
    assert torch.allclose(
        loss_token_no_mask, loss_seq_no_mask, rtol=1e-5
    ), "token_mean and sequence_mean should give same results when sequences have equal length and no mask"
    # But they should give different results when mask creates different effective sequence lengths
    assert not torch.allclose(
        loss_token_with_mask, loss_seq_with_mask, rtol=1e-3
    ), "token_mean and sequence_mean with mask should give different results"


def test_policy_loss_reduction_edge_cases():
    """Tests edge cases for loss_reduction modes."""

    device = "cpu"

    # Test with single sequence (should give same result for both modes)
    advantages = torch.tensor([[1.0, -1.0, 2.0]], device=device)
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    log_probs = torch.tensor([[-1.5, -0.5, -1.2]], device=device)

    # Create configs for different reduction modes
    config_token = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    config_seq = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "sequence_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    # Get loss function
    loss_fn = PolicyLossRegistry.get("regular")

    loss_token, _ = loss_fn(log_probs, old_log_probs, advantages, config_token)
    loss_seq, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq)

    # With single sequence, both modes should give same result
    torch.testing.assert_close(loss_token, loss_seq, rtol=1e-6, atol=1e-8)

    # Test with completely masked sequence
    loss_mask = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    loss_token_masked, _ = loss_fn(log_probs, old_log_probs, advantages, config_token, loss_mask)
    loss_seq_masked, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq, loss_mask)

    # Should handle zero mask gracefully (due to +1e-8 in denominator)
    assert torch.isfinite(loss_token_masked)
    assert torch.isfinite(loss_seq_masked)


def test_gspo_importance_sampling_levels():
    """Tests GSPO policy loss function with sequence-level importance sampling.

    This test focuses on GSPO's key benefit: stabilizing clipping behavior through sequence-level
    importance sampling, which should lead to more consistent training dynamics compared to
    token-level importance sampling in standard PPO.
    """

    device = "cpu"

    clip_eps_low = 0.2
    clip_eps_high = 0.2

    # Create test data with varied sequence lengths and extreme ratios to test clipping stability
    # GSPO's benefit is most apparent with sequences of different lengths and high variance
    advantages = torch.tensor(
        [
            [1.5, 2.0, 1.0, 0.8, 0.5, 0.0, 0.0, 0.0],  # long sequence: 5 valid tokens
            [3.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # short sequence: 2 valid tokens
            [0.5, 0.8, 1.2, 2.5, 0.0, 0.0, 0.0, 0.0],  # medium sequence: 4 valid tokens
        ],
        device=device,
    )

    old_log_probs = torch.tensor(
        [
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ],
        device=device,
    )

    # Create extreme log probability ratios to trigger significant clipping
    # This tests GSPO's stability benefits under conditions that would cause unstable clipping
    log_probs = torch.tensor(
        [
            [0.2, -2.5, -0.3, 0.1, -1.8, -1.0, -1.0, -1.0],  # high variance within sequence
            [0.8, -0.2, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # extreme ratios (exp(1.8)≈6.0, exp(0.8)≈2.2)
            [-0.5, 0.3, -1.7, 0.4, -1.0, -1.0, -1.0, -1.0],  # mixed extreme values
        ],
        device=device,
    )

    # Create masks for different sequence lengths (key for testing length normalization)
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # 5 tokens
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 tokens
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 4 tokens
        ],
        device=device,
    )

    # Test standard PPO (token-level importance sampling)
    ppo_config = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )
    ppo_loss_fn = PolicyLossRegistry.get("regular")
    loss_token, _ = ppo_loss_fn(log_probs, old_log_probs, advantages, ppo_config, loss_mask)

    # Test GSPO (sequence-level importance sampling)
    gspo_config = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "gspo",
            "loss_reduction": "sequence_mean",  # GSPO recommended reduction
            "max_seq_len": 4,
            "use_tis": False,
        }
    )
    gspo_loss_fn = PolicyLossRegistry.get("gspo")
    loss_sequence, _ = gspo_loss_fn(log_probs, old_log_probs, advantages, gspo_config, loss_mask)

    # Manual calculation for token-level (standard PPO)
    log_ratio = log_probs - old_log_probs
    ratio_token = log_ratio.exp()
    surr1_token = ratio_token * advantages
    surr2_token = ratio_token.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages
    loss_per_token_token = -torch.min(surr1_token, surr2_token)
    expected_token = (loss_per_token_token * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    # Calculate token-level clipping ratio
    is_clipped_token = (-surr2_token > -surr1_token) & (loss_mask.bool())
    clip_ratio_token = is_clipped_token.float().sum() / loss_mask.sum()

    # Manual calculation for sequence-level (GSPO)
    # First compute sequence-level importance weights (key GSPO innovation)
    log_importance_weights_seq = masked_mean(log_ratio, loss_mask, dim=-1).unsqueeze(-1)

    # GSPO uses stop gradients: s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_probs - sg[log_probs]
    ratio_sequence = torch.exp(log_importance_weights_seq.detach() + log_probs - log_probs.detach())
    surr1_sequence = ratio_sequence * advantages
    surr2_sequence = ratio_sequence.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages
    loss_per_token_sequence = -torch.min(surr1_sequence, surr2_sequence)
    # GSPO uses sequence_mean reduction
    expected_sequence = masked_mean(loss_per_token_sequence, loss_mask, dim=-1).mean()

    # Calculate sequence-level clipping ratio
    is_clipped_sequence = (-surr2_sequence > -surr1_sequence) & (loss_mask.bool())
    clip_ratio_sequence = is_clipped_sequence.float().sum() / loss_mask.sum()

    # Verify loss calculations
    torch.testing.assert_close(loss_token, expected_token, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_sequence, expected_sequence, rtol=1e-5, atol=1e-8)

    # Core GSPO benefit test: Different clipping behavior
    # GSPO should produce different clipping patterns due to sequence-level importance sampling
    assert not torch.allclose(
        clip_ratio_token, clip_ratio_sequence, rtol=1e-2
    ), f"Clipping ratios should differ: token={clip_ratio_token:.4f} vs sequence={clip_ratio_sequence:.4f}"

    # Test stability: sequence-level should smooth out extreme per-token variations
    # Check that sequence-level ratios have lower variance within each sequence
    token_ratio_variance = torch.var(ratio_token * loss_mask, dim=-1).mean()
    sequence_ratio_variance = torch.var(ratio_sequence * loss_mask, dim=-1).mean()

    # The key insight: GSPO should reduce within-sequence variance by using sequence-averaged ratios
    assert sequence_ratio_variance < token_ratio_variance, (
        f"GSPO should reduce ratio variance: sequence={sequence_ratio_variance:.4f} < "
        f"token={token_ratio_variance:.4f}"
    )

    # Token-level and sequence-level should give different results due to different importance weighting
    assert not torch.allclose(
        loss_token, loss_sequence, rtol=1e-3
    ), f"Loss values should differ: token={loss_token:.6f} vs sequence={loss_sequence:.6f}"

    # Test length normalization effect: sequences with different lengths should be handled more uniformly
    # This is a key stability benefit of GSPO mentioned in the paper
    seq_lengths = loss_mask.sum(dim=-1)  # [5, 2, 4]

    # In GSPO, the sequence-level importance weights should be the same across all tokens in a sequence
    # This should make the treatment more uniform across different sequence lengths
    for seq_idx in range(log_importance_weights_seq.shape[0]):
        seq_len = int(seq_lengths[seq_idx])
        if seq_len > 1:
            # All importance weights within a sequence should be identical (GSPO property)
            seq_weights = log_importance_weights_seq[seq_idx, :seq_len]
            assert torch.allclose(
                seq_weights, seq_weights[0], rtol=1e-6
            ), f"GSPO should have uniform importance weights within sequence {seq_idx}"


def test_clip_cov_policy_loss():
    """Tests Clip-Cov policy loss function with covariance-based correction."""

    device = "cpu"
    torch.manual_seed(42)  # For reproducible randomization in clip-cov

    # Create test data
    advantages = torch.tensor(
        [
            [2.0, -1.0, 1.5, 0.8],
            [1.0, 0.5, -2.0, 1.2],
        ],
        device=device,
    )

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]], device=device)

    log_probs = torch.tensor([[-0.5, -1.5, -0.8, -1.2], [-1.3, -0.7, -1.8, -0.9]], device=device)

    loss_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]], device=device)  # Last token masked

    # Create Clip-Cov config
    config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "policy_loss_type": "clip_cov",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "clip_cov": {"clip_ratio": 0.5, "clip_cov_lb": -5.0, "clip_cov_ub": 5.0},  # Large ratio for testing
        }
    )

    # Get loss function
    clip_cov_fn = PolicyLossRegistry.get("clip_cov")

    # Calculate loss
    loss, clip_frac = clip_cov_fn(log_probs, old_log_probs, advantages, config, loss_mask)

    # Basic sanity checks
    assert torch.isfinite(loss), "Loss should be finite"
    assert 0 <= clip_frac <= 1, f"Clip fraction should be between 0 and 1, got {clip_frac}"

    # Compare with regular PPO (should be different due to covariance correction)
    regular_config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    regular_fn = PolicyLossRegistry.get("regular")
    regular_loss, regular_clip_frac = regular_fn(log_probs, old_log_probs, advantages, regular_config, loss_mask)

    # Clip-Cov should give different results due to covariance-based correction
    assert not torch.allclose(
        loss, regular_loss, rtol=1e-3
    ), f"Clip-Cov and regular PPO should differ: clip_cov={loss:.6f} vs regular={regular_loss:.6f}"


def test_kl_cov_policy_loss():
    """Tests KL-Cov policy loss function with covariance-based token selection."""

    device = "cpu"
    torch.manual_seed(42)  # For reproducible token selection

    # Create test data
    advantages = torch.tensor(
        [
            [1.5, -0.5, 2.0, 0.8],
            [0.5, 1.0, -1.5, 1.2],
        ],
        device=device,
    )

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]], device=device)

    log_probs = torch.tensor([[-0.8, -1.2, -0.6, -1.1], [-1.1, -0.9, -1.4, -0.7]], device=device)

    loss_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]], device=device)  # Last token masked

    # Create KL-Cov config
    config = DictConfig(
        {
            "policy_loss_type": "kl_cov",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "kl_cov": {"kl_cov_frac": 0.5, "ppo_kl_coef": 1.0},  # Apply KL to 50% of tokens
        }
    )

    # Get loss function
    kl_cov_fn = PolicyLossRegistry.get("kl_cov")

    # Calculate loss
    loss, clip_frac = kl_cov_fn(log_probs, old_log_probs, advantages, config, loss_mask)

    # Basic sanity checks
    assert torch.isfinite(loss), "Loss should be finite"
    assert clip_frac == 0.0, "KL-Cov should return 0.0 for clipfrac value"

    # Compare with regular PPO (should be different due to KL regularization)
    regular_config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    regular_fn = PolicyLossRegistry.get("regular")
    regular_loss, _ = regular_fn(log_probs, old_log_probs, advantages, regular_config, loss_mask)

    # KL-Cov should give different results due to KL regularization on selected tokens
    assert not torch.allclose(
        loss, regular_loss, rtol=1e-3
    ), f"KL-Cov and regular PPO should differ: kl_cov={loss:.6f} vs regular={regular_loss:.6f}"


def test_branched_grpo_prefix_trie():
    """Tests prefix trie construction and localized statistics for branched GRPO.

    This test verifies the core mechanism of branched GRPO: building a prefix trie
    over trajectories and computing localized mean/std for tokens in shared prefixes.

    Example with branching:
        traj_1: [A, B, C, D, E] → reward r1
        traj_2: [A, B, C, X, Y] → reward r2 (branched from traj_1 at token C)
        traj_3: [A, B, C, X, Z] → reward r3 (branched from traj_2 at token X)

    Localized advantages:
        - Tokens A, B, C: use mean([r1, r2, r3]), std([r1, r2, r3]) — 3 trajectories share this prefix
        - Token D, E: use group mean/std (only traj_1 continues here)
        - Token X: use mean([r2, r3]), std([r2, r3]) — 2 trajectories share this prefix
        - Token Y: use group mean/std (only traj_2)
        - Token Z: use group mean/std (only traj_3)
    """

    device = "cpu"

    # Create test data: 3 trajectories with branching structure
    # traj_1: [1, 2, 3, 4, 5] → reward 1.0
    # traj_2: [1, 2, 3, 6, 7] → reward 0.5 (branched at token 3)
    # traj_3: [1, 2, 3, 6, 8] → reward 0.0 (branched at token 6)
    sequences = torch.tensor(
        [
            [1, 2, 3, 4, 5, 0, 0],  # traj_1
            [1, 2, 3, 6, 7, 0, 0],  # traj_2
            [1, 2, 3, 6, 8, 0, 0],  # traj_3
        ],
        device=device,
    )

    rewards = torch.tensor([1.0, 0.5, 0.0], device=device)

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
        ],
        device=device,
    )

    # Build prefix trie
    trie_root = build_prefix_trie(sequences, rewards, attention_mask)

    # Verify trie structure: tokens 1, 2, 3 should have all 3 trajectories
    node = trie_root
    for token_id in [1, 2, 3]:
        node = node.children[token_id]
        assert len(node.trajectory_rewards) == 3, f"Token {token_id} should have 3 trajectories"
        assert set(node.trajectory_rewards) == {1.0, 0.5, 0.0}

    # Token 4 should have only traj_1
    node_after_3 = trie_root.children[1].children[2].children[3]
    assert 4 in node_after_3.children
    node_4 = node_after_3.children[4]
    assert len(node_4.trajectory_rewards) == 1
    assert node_4.trajectory_rewards[0] == 1.0

    # Token 6 should have traj_2 and traj_3
    assert 6 in node_after_3.children
    node_6 = node_after_3.children[6]
    assert len(node_6.trajectory_rewards) == 2
    assert set(node_6.trajectory_rewards) == {0.5, 0.0}

    # Compute localized stats
    group_mean = rewards.mean().item()  # 0.5
    group_std = rewards.std().item()  # 0.5

    localized_means, localized_stds = compute_localized_stats(
        trie_root, sequences, attention_mask, group_mean, group_std
    )

    # Hand-calculated expected values:
    # Positions 0, 1, 2 (tokens 1, 2, 3): all 3 trajectories share
    #   mean = (1.0 + 0.5 + 0.0) / 3 = 0.5
    #   std = sqrt(((1.0-0.5)^2 + (0.5-0.5)^2 + (0.0-0.5)^2) / 3) ≈ 0.4082
    expected_mean_3_trajs = 0.5
    expected_std_3_trajs = torch.tensor([1.0, 0.5, 0.0]).std().item()  # ≈ 0.5 (sample std)

    # Position 3 for traj_1 (token 4): only 1 trajectory → falls back to group stats
    # Position 3 for traj_2, traj_3 (token 6): 2 trajectories share
    #   mean = (0.5 + 0.0) / 2 = 0.25
    #   std = sqrt(((0.5-0.25)^2 + (0.0-0.25)^2) / 2) ≈ 0.3536
    expected_mean_2_trajs = 0.25
    expected_std_2_trajs = torch.tensor([0.5, 0.0]).std().item()  # ≈ 0.3536 (sample std)

    # Verify localized means for positions 0, 1, 2 (shared by all 3)
    for pos in range(3):
        for traj_idx in range(3):
            assert localized_means[traj_idx, pos].item() == pytest.approx(expected_mean_3_trajs, abs=1e-5)
            assert localized_stds[traj_idx, pos].item() == pytest.approx(expected_std_3_trajs, abs=1e-5)

    # Verify position 3 for traj_1 (token 4) uses group stats
    assert localized_means[0, 3].item() == pytest.approx(group_mean, abs=1e-5)
    assert localized_stds[0, 3].item() == pytest.approx(group_std, abs=1e-5)

    # Verify position 3 for traj_2, traj_3 (token 6) uses localized stats from 2 trajectories
    assert localized_means[1, 3].item() == pytest.approx(expected_mean_2_trajs, abs=1e-5)
    assert localized_means[2, 3].item() == pytest.approx(expected_mean_2_trajs, abs=1e-5)
    assert localized_stds[1, 3].item() == pytest.approx(expected_std_2_trajs, abs=1e-5)
    assert localized_stds[2, 3].item() == pytest.approx(expected_std_2_trajs, abs=1e-5)


def test_branched_grpo_advantage_computation():
    """Tests the full branched GRPO advantage computation.

    Verifies that advantages are computed correctly using localized statistics
    at branch points, with proper masking for model-generated tokens.
    """

    device = "cpu"

    # Create test data: 3 trajectories with branching structure
    sequences = torch.tensor(
        [
            [1, 2, 3, 4, 5, 0, 0],  # traj_1
            [1, 2, 3, 6, 7, 0, 0],  # traj_2
            [1, 2, 3, 6, 8, 0, 0],  # traj_3
        ],
        device=device,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
        ],
        device=device,
    )

    # Token-level rewards: outcome reward placed at last valid token (position 4)
    token_level_rewards = torch.zeros(3, 7, device=device, dtype=torch.float32)
    token_level_rewards[0, 4] = 1.0  # traj_1 reward
    token_level_rewards[1, 4] = 0.5  # traj_2 reward
    token_level_rewards[2, 4] = 0.0  # traj_3 reward

    response_mask = attention_mask.clone().float()
    loss_mask = attention_mask.clone().float()  # All tokens are model-generated

    index = np.array(["uid1", "uid1", "uid1"])

    # Compute branched GRPO advantages
    advantages, returns = compute_branched_grpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        sequences=sequences,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        grpo_norm_by_std=True,
    )

    # Verify output shapes
    assert advantages.shape == (3, 7)
    assert returns.shape == (3, 7)

    # Verify padding positions have zero advantage
    assert torch.all(advantages[:, 5:] == 0)

    # Hand-calculated expected advantages for position 0 (shared by all 3):
    # outcome_reward = [1.0, 0.5, 0.0]
    # localized_mean = 0.5, localized_std = 0.5
    # advantage = (outcome - mean) / (std + eps)
    # traj_1: (1.0 - 0.5) / (0.5 + 1e-6) ≈ 1.0
    # traj_2: (0.5 - 0.5) / (0.5 + 1e-6) ≈ 0.0
    # traj_3: (0.0 - 0.5) / (0.5 + 1e-6) ≈ -1.0
    assert advantages[0, 0].item() == pytest.approx(1.0, abs=1e-4)
    assert advantages[1, 0].item() == pytest.approx(0.0, abs=1e-4)
    assert advantages[2, 0].item() == pytest.approx(-1.0, abs=1e-4)

    # Position 3 for traj_2, traj_3 (token 6) uses localized stats from 2 trajectories:
    # localized_mean = 0.25, localized_std ≈ 0.3536
    # traj_2: (0.5 - 0.25) / (0.3536 + 1e-6) ≈ 0.707
    # traj_3: (0.0 - 0.25) / (0.3536 + 1e-6) ≈ -0.707
    assert advantages[1, 3].item() == pytest.approx(0.707, abs=1e-2)
    assert advantages[2, 3].item() == pytest.approx(-0.707, abs=1e-2)


def test_branched_grpo_multiple_uids():
    """Tests branched GRPO with multiple UID groups."""

    device = "cpu"

    # Create test data: 2 UID groups, each with 2 trajectories
    sequences = torch.tensor(
        [
            [1, 2, 3, 0],  # uid1, traj_1
            [1, 2, 4, 0],  # uid1, traj_2 (branched at token 2)
            [5, 6, 7, 0],  # uid2, traj_1
            [5, 6, 8, 0],  # uid2, traj_2 (branched at token 6)
        ],
        device=device,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ],
        device=device,
    )

    token_level_rewards = torch.zeros(4, 4, device=device, dtype=torch.float32)
    token_level_rewards[0, 2] = 1.0  # uid1, traj_1
    token_level_rewards[1, 2] = 0.0  # uid1, traj_2
    token_level_rewards[2, 2] = 0.8  # uid2, traj_1
    token_level_rewards[3, 2] = 0.2  # uid2, traj_2

    response_mask = attention_mask.clone().float()
    loss_mask = attention_mask.clone().float()

    index = np.array(["uid1", "uid1", "uid2", "uid2"])

    advantages, _ = compute_branched_grpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        sequences=sequences,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        grpo_norm_by_std=True,
    )

    # Verify UID groups are processed independently
    # uid1: positions 0, 1 (tokens 1, 2) are shared, position 2 is not
    # uid2: positions 0, 1 (tokens 5, 6) are shared, position 2 is not

    # uid1 shared prefix (positions 0, 1): mean=0.5, std=0.707
    # uid1 traj_1: (1.0 - 0.5) / (0.707 + 1e-6) ≈ 0.707
    # uid1 traj_2: (0.0 - 0.5) / (0.707 + 1e-6) ≈ -0.707
    assert advantages[0, 0].item() == pytest.approx(0.707, abs=1e-2)
    assert advantages[1, 0].item() == pytest.approx(-0.707, abs=1e-2)

    # uid2 shared prefix (positions 0, 1): mean=0.5, std=0.424
    # uid2 traj_1: (0.8 - 0.5) / (0.424 + 1e-6) ≈ 0.707
    # uid2 traj_2: (0.2 - 0.5) / (0.424 + 1e-6) ≈ -0.707
    assert advantages[2, 0].item() == pytest.approx(0.707, abs=1e-2)
    assert advantages[3, 0].item() == pytest.approx(-0.707, abs=1e-2)


def test_branched_grpo_no_branching():
    """Tests branched GRPO when trajectories have no shared prefixes.

    When all trajectories diverge immediately, branched GRPO should behave
    identically to standard GRPO (using group-level statistics everywhere).
    """

    device = "cpu"

    # Create test data: 3 trajectories with no shared prefix
    sequences = torch.tensor(
        [
            [1, 2, 3, 0],  # traj_1
            [4, 5, 6, 0],  # traj_2
            [7, 8, 9, 0],  # traj_3
        ],
        device=device,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ],
        device=device,
    )

    token_level_rewards = torch.zeros(3, 4, device=device, dtype=torch.float32)
    token_level_rewards[0, 2] = 1.0
    token_level_rewards[1, 2] = 0.5
    token_level_rewards[2, 2] = 0.0

    response_mask = attention_mask.clone().float()
    loss_mask = attention_mask.clone().float()

    index = np.array(["uid1", "uid1", "uid1"])

    advantages, _ = compute_branched_grpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        sequences=sequences,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        grpo_norm_by_std=True,
    )

    # With no shared prefixes, all positions should use group-level stats
    # group_mean = 0.5, group_std = 0.5
    # All positions for each trajectory should have the same advantage
    expected_adv_traj1 = (1.0 - 0.5) / (0.5 + 1e-6)  # ≈ 1.0
    expected_adv_traj2 = (0.5 - 0.5) / (0.5 + 1e-6)  # ≈ 0.0
    expected_adv_traj3 = (0.0 - 0.5) / (0.5 + 1e-6)  # ≈ -1.0

    for pos in range(3):
        assert advantages[0, pos].item() == pytest.approx(expected_adv_traj1, abs=1e-4)
        assert advantages[1, pos].item() == pytest.approx(expected_adv_traj2, abs=1e-4)
        assert advantages[2, pos].item() == pytest.approx(expected_adv_traj3, abs=1e-4)


def test_branched_grpo_trainer_shapes():
    """Tests branched GRPO with shapes matching actual trainer data.

    In the trainer:
    - sequences: [batch, prompt_len + response_len] (full trajectory)
    - attention_mask: [batch, prompt_len + response_len] (full trajectory)
    - response_mask: [batch, response_len] (response only)
    - loss_mask: [batch, response_len] (response only)
    - token_level_rewards: [batch, response_len] (response only)
    """
    device = "cpu"

    # Simulate trainer shapes: prompt_len=3, response_len=5
    prompt_len = 3
    response_len = 5
    batch_size = 3

    # Full sequences (prompt + response)
    # Prompt tokens: [100, 101, 102] (same for all)
    # Response tokens vary to create branching
    sequences = torch.tensor(
        [
            [100, 101, 102, 1, 2, 3, 4, 5],  # prompt + response_1
            [100, 101, 102, 1, 2, 3, 6, 7],  # prompt + response_2 (branches at pos 3)
            [100, 101, 102, 1, 2, 3, 6, 8],  # prompt + response_3 (branches at pos 4)
        ],
        device=device,
    )

    # Full attention mask
    attention_mask = torch.ones(batch_size, prompt_len + response_len, device=device, dtype=torch.int64)

    # Response-only masks
    response_mask = torch.ones(batch_size, response_len, device=device, dtype=torch.float32)
    loss_mask = torch.ones(batch_size, response_len, device=device, dtype=torch.float32)

    # Token-level rewards (response only)
    token_level_rewards = torch.zeros(batch_size, response_len, device=device, dtype=torch.float32)
    token_level_rewards[0, -1] = 1.0  # reward at last position
    token_level_rewards[1, -1] = 0.5
    token_level_rewards[2, -1] = 0.0

    index = ["uid1", "uid1", "uid1"]  # List[str] like actual trainer

    # Compute advantages
    advantages, returns = compute_branched_grpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        sequences=sequences,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        grpo_norm_by_std=True,
    )

    # Verify output shapes match response_mask (not sequences)
    assert advantages.shape == (batch_size, response_len)
    assert returns.shape == (batch_size, response_len)

    # Verify advantages are computed correctly
    # Position 0-2 in response (tokens 1,2,3) shared by all 3 trajectories
    # localized_mean = 0.5, localized_std = 0.5
    assert advantages[0, 0].item() == pytest.approx(1.0, abs=1e-4)  # (1.0 - 0.5) / 0.5
    assert advantages[1, 0].item() == pytest.approx(0.0, abs=1e-4)  # (0.5 - 0.5) / 0.5
    assert advantages[2, 0].item() == pytest.approx(-1.0, abs=1e-4)  # (0.0 - 0.5) / 0.5

    # Position 3 (token 6) shared by traj_2 and traj_3 only
    # localized_mean = 0.25, localized_std ≈ 0.3536
    assert advantages[1, 3].item() == pytest.approx(0.707, abs=1e-2)
    assert advantages[2, 3].item() == pytest.approx(-0.707, abs=1e-2)


def test_branched_grpo_multi_turn_loss_mask():
    """Tests branched GRPO with multi-turn loss mask (env tokens have loss_mask=0).

    In multi-turn, loss_mask distinguishes model-generated tokens (1) from
    env observation tokens (0). Advantages should be zero for env tokens.
    """
    device = "cpu"

    prompt_len = 2
    response_len = 8
    batch_size = 2

    # Full sequences
    sequences = torch.tensor(
        [
            [100, 101, 1, 2, 50, 51, 3, 4, 0, 0],  # model, model, env, env, model, model, pad, pad
            [100, 101, 1, 2, 50, 51, 5, 6, 0, 0],  # model, model, env, env, model, model, pad, pad
        ],
        device=device,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ],
        device=device,
    )

    # Response-only: model tokens have loss_mask=1, env tokens have loss_mask=0
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0],  # valid response tokens
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        device=device,
        dtype=torch.float32,
    )

    loss_mask = torch.tensor(
        [
            [1, 1, 0, 0, 1, 1, 0, 0],  # 1=model, 0=env or pad
            [1, 1, 0, 0, 1, 1, 0, 0],
        ],
        device=device,
        dtype=torch.float32,
    )

    token_level_rewards = torch.zeros(batch_size, response_len, device=device, dtype=torch.float32)
    token_level_rewards[0, 5] = 1.0
    token_level_rewards[1, 5] = 0.0

    index = ["uid1", "uid1"]

    advantages, _ = compute_branched_grpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        sequences=sequences,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        grpo_norm_by_std=True,
    )

    assert advantages.shape == (batch_size, response_len)

    # Env token positions (2, 3) should have zero advantage
    assert advantages[0, 2].item() == 0.0
    assert advantages[0, 3].item() == 0.0
    assert advantages[1, 2].item() == 0.0
    assert advantages[1, 3].item() == 0.0

    # Padding positions should have zero advantage
    assert advantages[0, 6].item() == 0.0
    assert advantages[0, 7].item() == 0.0

    # Model token positions should have non-zero advantage
    # (unless normalized advantage happens to be 0)
    # traj_1: reward=1.0, traj_2: reward=0.0, mean=0.5, std=0.707
    # traj_1 advantage: (1.0 - 0.5) / 0.707 ≈ 0.707
    # traj_2 advantage: (0.0 - 0.5) / 0.707 ≈ -0.707
    assert advantages[0, 0].item() == pytest.approx(0.707, abs=1e-2)
    assert advantages[1, 0].item() == pytest.approx(-0.707, abs=1e-2)
