"""Tests for model implementations."""

import numpy as np
import pytest
import torch

from offline_robotic_manipulation_curriculum.models.model import (
    QNetwork,
    PolicyNetwork,
    EnsembleQNetwork,
    BehavioralCloningAgent,
    CQLAgent,
)


class TestQNetwork:
    """Tests for Q-network."""

    def test_initialization(self, device):
        """Test Q-network initialization."""
        obs_dim = 24
        act_dim = 4

        q_net = QNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=[128, 128],
            activation="relu",
        ).to(device)

        assert q_net.obs_dim == obs_dim
        assert q_net.act_dim == act_dim

    def test_forward_pass(self, device):
        """Test Q-network forward pass."""
        obs_dim = 24
        act_dim = 4
        batch_size = 32

        q_net = QNetwork(obs_dim, act_dim).to(device)

        obs = torch.randn(batch_size, obs_dim).to(device)
        actions = torch.randn(batch_size, act_dim).to(device)

        q_values = q_net(obs, actions)

        assert q_values.shape == (batch_size, 1)

    def test_different_activations(self, device):
        """Test Q-network with different activations."""
        obs_dim = 24
        act_dim = 4

        for activation in ["relu", "tanh", "elu"]:
            q_net = QNetwork(obs_dim, act_dim, activation=activation).to(device)
            obs = torch.randn(1, obs_dim).to(device)
            actions = torch.randn(1, act_dim).to(device)
            q_values = q_net(obs, actions)
            assert q_values.shape == (1, 1)


class TestPolicyNetwork:
    """Tests for policy network."""

    def test_initialization(self, device):
        """Test policy network initialization."""
        obs_dim = 24
        act_dim = 4

        policy = PolicyNetwork(obs_dim, act_dim).to(device)

        assert policy.obs_dim == obs_dim
        assert policy.act_dim == act_dim

    def test_forward_deterministic(self, device):
        """Test deterministic action selection."""
        obs_dim = 24
        act_dim = 4
        batch_size = 32

        policy = PolicyNetwork(obs_dim, act_dim).to(device)
        obs = torch.randn(batch_size, obs_dim).to(device)

        actions, log_probs = policy(obs, deterministic=True, with_logprob=False)

        assert actions.shape == (batch_size, act_dim)
        assert log_probs is None
        assert torch.all(torch.abs(actions) <= 1.0)

    def test_forward_stochastic(self, device):
        """Test stochastic action selection."""
        obs_dim = 24
        act_dim = 4
        batch_size = 32

        policy = PolicyNetwork(obs_dim, act_dim).to(device)
        obs = torch.randn(batch_size, obs_dim).to(device)

        actions, log_probs = policy(obs, deterministic=False, with_logprob=True)

        assert actions.shape == (batch_size, act_dim)
        assert log_probs.shape == (batch_size, 1)
        assert torch.all(torch.abs(actions) <= 1.0)


class TestEnsembleQNetwork:
    """Tests for ensemble Q-network."""

    def test_initialization(self, device):
        """Test ensemble initialization."""
        obs_dim = 24
        act_dim = 4
        num_networks = 5

        ensemble = EnsembleQNetwork(
            obs_dim, act_dim, num_networks=num_networks
        ).to(device)

        assert len(ensemble.networks) == num_networks

    def test_forward_pass(self, device):
        """Test ensemble forward pass."""
        obs_dim = 24
        act_dim = 4
        batch_size = 32
        num_networks = 5

        ensemble = EnsembleQNetwork(
            obs_dim, act_dim, num_networks=num_networks
        ).to(device)

        obs = torch.randn(batch_size, obs_dim).to(device)
        actions = torch.randn(batch_size, act_dim).to(device)

        q_values = ensemble(obs, actions)

        assert q_values.shape == (batch_size, num_networks)

    def test_uncertainty_estimation(self, device):
        """Test epistemic uncertainty estimation."""
        obs_dim = 24
        act_dim = 4
        batch_size = 32

        ensemble = EnsembleQNetwork(obs_dim, act_dim, num_networks=5).to(device)

        obs = torch.randn(batch_size, obs_dim).to(device)
        actions = torch.randn(batch_size, act_dim).to(device)

        uncertainty = ensemble.get_uncertainty(obs, actions)

        assert uncertainty.shape == (batch_size,)
        assert torch.all(uncertainty >= 0.0)


class TestBehavioralCloningAgent:
    """Tests for behavioral cloning agent."""

    def test_initialization(self, device):
        """Test BC agent initialization."""
        obs_dim = 24
        act_dim = 4

        bc_agent = BehavioralCloningAgent(obs_dim, act_dim).to(device)

        assert bc_agent.policy is not None

    def test_forward_pass(self, device):
        """Test BC agent forward pass."""
        obs_dim = 24
        act_dim = 4
        batch_size = 32

        bc_agent = BehavioralCloningAgent(obs_dim, act_dim).to(device)

        obs = torch.randn(batch_size, obs_dim).to(device)
        actions = bc_agent(obs, deterministic=True)

        assert actions.shape == (batch_size, act_dim)

    def test_loss_computation(self, device, sample_batch):
        """Test BC loss computation."""
        obs_dim = sample_batch["observations"].shape[1]
        act_dim = sample_batch["actions"].shape[1]

        bc_agent = BehavioralCloningAgent(obs_dim, act_dim).to(device)

        loss = bc_agent.compute_loss(
            sample_batch["observations"],
            sample_batch["actions"],
        )

        assert loss.ndim == 0  # Scalar loss
        assert loss.item() >= 0.0


class TestCQLAgent:
    """Tests for CQL agent."""

    def test_initialization(self, device):
        """Test CQL agent initialization."""
        obs_dim = 24
        act_dim = 4

        agent = CQLAgent(obs_dim, act_dim, ensemble_size=3).to(device)

        assert agent.q_ensemble is not None
        assert agent.target_q_ensemble is not None
        assert agent.policy is not None

    def test_select_action(self, device):
        """Test action selection."""
        obs_dim = 24
        act_dim = 4

        agent = CQLAgent(obs_dim, act_dim).to(device)

        obs = np.random.randn(obs_dim)
        action = agent.select_action(obs, deterministic=True)

        assert action.shape == (act_dim,)
        assert np.all(np.abs(action) <= 1.0)

    def test_cql_loss_computation(self, device, sample_batch):
        """Test CQL loss computation."""
        obs_dim = sample_batch["observations"].shape[1]
        act_dim = sample_batch["actions"].shape[1]

        agent = CQLAgent(obs_dim, act_dim, ensemble_size=3).to(device)

        loss_dict = agent.compute_cql_loss(
            obs=sample_batch["observations"],
            actions=sample_batch["actions"],
            rewards=sample_batch["rewards"],
            next_obs=sample_batch["next_observations"],
            terminals=sample_batch["terminals"],
            gamma=0.99,
        )

        assert "total_loss" in loss_dict
        assert "bellman_loss" in loss_dict
        assert "cql_penalty" in loss_dict
        assert "current_q" in loss_dict

        assert loss_dict["total_loss"].ndim == 0

    def test_target_network_update(self, device):
        """Test soft target network update."""
        obs_dim = 24
        act_dim = 4

        agent = CQLAgent(obs_dim, act_dim).to(device)

        # Get initial target parameters
        initial_params = [
            p.clone() for p in agent.target_q_ensemble.parameters()
        ]

        # Modify Q-network parameters
        for p in agent.q_ensemble.parameters():
            p.data += 1.0

        # Update target networks
        agent.update_target_networks(tau=0.5)

        # Check that target parameters changed
        updated_params = list(agent.target_q_ensemble.parameters())
        for initial, updated in zip(initial_params, updated_params):
            assert not torch.allclose(initial, updated)

    def test_lagrange_multiplier(self, device):
        """Test CQL with Lagrange multiplier."""
        obs_dim = 24
        act_dim = 4

        agent = CQLAgent(
            obs_dim, act_dim, use_lagrange=True, ensemble_size=3
        ).to(device)

        assert hasattr(agent, "log_alpha_lagrange")
        assert agent.log_alpha_lagrange.requires_grad
