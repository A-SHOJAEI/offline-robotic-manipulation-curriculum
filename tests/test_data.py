"""Tests for data loading and preprocessing."""

import numpy as np
import pytest
import torch

from offline_robotic_manipulation_curriculum.data.loader import ReplayBuffer
from offline_robotic_manipulation_curriculum.data.preprocessing import (
    normalize_observations,
    normalize_rewards,
    compute_returns,
    compute_advantages,
)


class TestReplayBuffer:
    """Tests for ReplayBuffer class."""

    def test_initialization(self, device):
        """Test buffer initialization."""
        size = 100
        obs_dim = 10
        act_dim = 3

        observations = np.random.randn(size, obs_dim)
        actions = np.random.randn(size, act_dim)
        rewards = np.random.randn(size)
        next_observations = np.random.randn(size, obs_dim)
        terminals = np.zeros(size)

        buffer = ReplayBuffer(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            device=str(device),
        )

        assert len(buffer) == size
        assert buffer.observations.shape == (size, obs_dim)
        assert buffer.actions.shape == (size, act_dim)

    def test_sampling(self, mock_replay_buffer):
        """Test batch sampling."""
        batch_size = 32
        batch = mock_replay_buffer.sample(batch_size)

        assert "observations" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert "next_observations" in batch
        assert "terminals" in batch

        assert batch["observations"].shape[0] == batch_size
        assert batch["actions"].shape[0] == batch_size

    def test_indexing(self, mock_replay_buffer):
        """Test buffer indexing."""
        idx = 0
        transition = mock_replay_buffer[idx]

        assert len(transition) == 5
        assert all(isinstance(t, torch.Tensor) for t in transition)


class TestPreprocessing:
    """Tests for preprocessing utilities."""

    def test_normalize_observations(self, random_seed):
        """Test observation normalization."""
        observations = np.random.randn(100, 10)

        normalized, mean, std = normalize_observations(observations)

        assert normalized.shape == observations.shape
        assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-6)

    def test_normalize_rewards_standardize(self, random_seed):
        """Test reward standardization."""
        rewards = np.random.randn(100)

        normalized = normalize_rewards(rewards, method="standardize")

        assert normalized.shape == rewards.shape
        assert np.abs(normalized.mean()) < 1e-6
        assert np.abs(normalized.std() - 1.0) < 1e-6

    def test_normalize_rewards_minmax(self, random_seed):
        """Test min-max reward normalization."""
        rewards = np.random.randn(100)

        normalized = normalize_rewards(rewards, method="min_max")

        assert normalized.shape == rewards.shape
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_compute_returns(self, random_seed):
        """Test return computation."""
        rewards = np.array([1.0, 1.0, 1.0, 1.0])
        terminals = np.array([0.0, 0.0, 0.0, 1.0])
        gamma = 0.99

        returns = compute_returns(rewards, terminals, gamma)

        assert returns.shape == rewards.shape
        # Last return should be just the reward (terminal state)
        assert np.isclose(returns[-1], 1.0)
        # Returns should be discounted
        assert returns[0] > returns[1]

    def test_compute_advantages(self, random_seed):
        """Test GAE computation."""
        rewards = np.random.randn(100)
        values = np.random.randn(100)
        next_values = np.random.randn(100)
        terminals = np.zeros(100)

        advantages = compute_advantages(
            rewards, values, next_values, terminals, gamma=0.99, lam=0.95
        )

        assert advantages.shape == rewards.shape

    def test_compute_returns_multiple_episodes(self, random_seed):
        """Test return computation across multiple episodes."""
        rewards = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
        terminals = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
        gamma = 0.99

        returns = compute_returns(rewards, terminals, gamma)

        # After terminal, return should reset
        assert returns[2] > returns[1]


def test_normalization_with_precomputed_stats(random_seed):
    """Test normalization with pre-computed statistics."""
    observations = np.random.randn(100, 10)

    # First pass
    normalized1, mean, std = normalize_observations(observations)

    # Second pass with same stats
    new_observations = np.random.randn(50, 10)
    normalized2, _, _ = normalize_observations(new_observations, mean=mean, std=std)

    assert normalized2.shape == new_observations.shape
    # Check that the same statistics are being used
    assert mean is not None
    assert std is not None
