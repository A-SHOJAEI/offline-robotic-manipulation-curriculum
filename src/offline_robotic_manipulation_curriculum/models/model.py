"""Model implementations for Conservative Q-Learning and Behavioral Cloning."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from offline_robotic_manipulation_curriculum.utils.nn_utils import get_activation

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Q-network for continuous action spaces."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """Initialize Q-network.

        Args:
            obs_dim: Observation dimension.
            act_dim: Action dimension.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function name.
            dropout: Dropout probability.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Build network
        layers = []
        input_dim = obs_dim + act_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network.

        Args:
            obs: Observations [batch_size, obs_dim].
            action: Actions [batch_size, act_dim].

        Returns:
            Q-values [batch_size, 1].
        """
        x = torch.cat([obs, action], dim=-1)
        return self.network(x)


class PolicyNetwork(nn.Module):
    """Policy network for continuous action spaces."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        """Initialize policy network.

        Args:
            obs_dim: Observation dimension.
            act_dim: Action dimension.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function name.
            dropout: Dropout probability.
            log_std_min: Minimum log standard deviation.
            log_std_max: Maximum log standard deviation.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build network
        layers = []
        input_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, act_dim)
        self.log_std_head = nn.Linear(input_dim, act_dim)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through policy network.

        Args:
            obs: Observations [batch_size, obs_dim].
            deterministic: If True, return mean action.
            with_logprob: If True, return log probabilities.

        Returns:
            Tuple of (actions, log_probs).
        """
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None if not with_logprob else torch.zeros_like(mean)
        else:
            dist = Normal(mean, std)
            sample = dist.rsample()
            action = torch.tanh(sample)

            if with_logprob:
                log_prob = dist.log_prob(sample)
                # Correction for tanh squashing
                log_prob -= torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                log_prob = None

        return action, log_prob


class EnsembleQNetwork(nn.Module):
    """Ensemble of Q-networks for uncertainty estimation."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_networks: int = 5,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """Initialize ensemble Q-network.

        Args:
            obs_dim: Observation dimension.
            act_dim: Action dimension.
            num_networks: Number of Q-networks in ensemble.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function name.
            dropout: Dropout probability.
        """
        super().__init__()

        self.num_networks = num_networks
        self.networks = nn.ModuleList([
            QNetwork(obs_dim, act_dim, hidden_dims, activation, dropout)
            for _ in range(num_networks)
        ])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble.

        Args:
            obs: Observations [batch_size, obs_dim].
            action: Actions [batch_size, act_dim].

        Returns:
            Q-values from all networks [batch_size, num_networks].
        """
        q_values = torch.stack([net(obs, action) for net in self.networks], dim=-1)
        return q_values.squeeze(1)

    def get_uncertainty(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute epistemic uncertainty as ensemble standard deviation.

        Args:
            obs: Observations [batch_size, obs_dim].
            action: Actions [batch_size, act_dim].

        Returns:
            Uncertainty estimates [batch_size].
        """
        q_values = self.forward(obs, action)
        uncertainty = q_values.std(dim=-1)
        return uncertainty


class BehavioralCloningAgent(nn.Module):
    """Behavioral cloning agent for warm-starting policies."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        """Initialize behavioral cloning agent.

        Args:
            obs_dim: Observation dimension.
            act_dim: Action dimension.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function name.
            dropout: Dropout probability.
        """
        super().__init__()

        self.policy = PolicyNetwork(
            obs_dim, act_dim, hidden_dims, activation, dropout
        )

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action from observation.

        Args:
            obs: Observations [batch_size, obs_dim].
            deterministic: If True, return mean action.

        Returns:
            Actions [batch_size, act_dim].
        """
        action, _ = self.policy(obs, deterministic=deterministic, with_logprob=False)
        return action

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute behavioral cloning loss.

        Args:
            obs: Observations [batch_size, obs_dim].
            actions: Expert actions [batch_size, act_dim].

        Returns:
            BC loss scalar.
        """
        pred_actions, _ = self.policy(obs, deterministic=False, with_logprob=False)
        loss = F.mse_loss(pred_actions, actions)
        return loss


class CQLAgent(nn.Module):
    """Conservative Q-Learning agent for offline RL."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = "relu",
        dropout: float = 0.1,
        ensemble_size: int = 5,
        cql_alpha: float = 5.0,
        min_q_weight: float = 5.0,
        use_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
    ):
        """Initialize CQL agent.

        Args:
            obs_dim: Observation dimension.
            act_dim: Action dimension.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function name.
            dropout: Dropout probability.
            ensemble_size: Number of Q-networks in ensemble.
            cql_alpha: CQL penalty coefficient.
            min_q_weight: Weight for min Q regularization.
            use_lagrange: Whether to use Lagrange multiplier for auto-tuning.
            lagrange_threshold: Target constraint value for Lagrange.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cql_alpha = cql_alpha
        self.min_q_weight = min_q_weight
        self.use_lagrange = use_lagrange
        self.lagrange_threshold = lagrange_threshold

        # Q-networks
        self.q_ensemble = EnsembleQNetwork(
            obs_dim, act_dim, ensemble_size, hidden_dims, activation, dropout
        )
        self.target_q_ensemble = EnsembleQNetwork(
            obs_dim, act_dim, ensemble_size, hidden_dims, activation, dropout
        )
        self.target_q_ensemble.load_state_dict(self.q_ensemble.state_dict())

        # Policy network
        self.policy = PolicyNetwork(obs_dim, act_dim, hidden_dims, activation, dropout)

        # Lagrange multiplier
        if use_lagrange:
            self.log_alpha_lagrange = nn.Parameter(torch.zeros(1))

        logger.info(f"Initialized CQL agent with ensemble size {ensemble_size}")

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action for given observation.

        Args:
            obs: Observation array.
            deterministic: If True, return mean action.

        Returns:
            Action array.
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(next(self.parameters()).device)
            action, _ = self.policy(obs_tensor, deterministic=deterministic, with_logprob=False)
            return action.cpu().numpy()[0]

    def compute_cql_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
    ) -> Dict[str, torch.Tensor]:
        """Compute CQL loss components.

        Args:
            obs: Observations [batch_size, obs_dim].
            actions: Actions [batch_size, act_dim].
            rewards: Rewards [batch_size].
            next_obs: Next observations [batch_size, obs_dim].
            terminals: Terminal flags [batch_size].
            gamma: Discount factor.

        Returns:
            Dictionary of loss components.
        """
        batch_size = obs.shape[0]

        # Current Q-values
        current_q = self.q_ensemble(obs, actions).mean(dim=-1)

        # Target Q-values
        with torch.no_grad():
            next_actions, next_log_probs = self.policy(next_obs, deterministic=False)
            target_q = self.target_q_ensemble(next_obs, next_actions).min(dim=-1)[0]
            target_q = rewards + gamma * (1 - terminals) * target_q

        # Bellman error
        bellman_loss = F.mse_loss(current_q, target_q)

        # CQL penalty: push down Q-values for OOD actions
        num_random_actions = 10
        random_actions = torch.FloatTensor(
            batch_size, num_random_actions, self.act_dim
        ).uniform_(-1, 1).to(obs.device)

        obs_repeated = obs.unsqueeze(1).repeat(1, num_random_actions, 1)
        obs_repeated = obs_repeated.view(-1, self.obs_dim)
        random_actions_flat = random_actions.view(-1, self.act_dim)

        random_q = self.q_ensemble(obs_repeated, random_actions_flat).mean(dim=-1)
        random_q = random_q.view(batch_size, num_random_actions)

        # Q-values under current policy
        policy_actions, policy_log_probs = self.policy(obs, deterministic=False)
        policy_q = self.q_ensemble(obs, policy_actions).mean(dim=-1)

        # CQL penalty
        logsumexp_random = torch.logsumexp(random_q, dim=1, keepdim=False)
        cql_penalty = (logsumexp_random - current_q).mean()

        # Total loss
        if self.use_lagrange:
            alpha_lagrange = torch.exp(self.log_alpha_lagrange).squeeze()
            cql_loss = bellman_loss + alpha_lagrange * (
                cql_penalty - self.lagrange_threshold
            )
        else:
            cql_loss = bellman_loss + self.cql_alpha * cql_penalty

        return {
            "total_loss": cql_loss,
            "bellman_loss": bellman_loss,
            "cql_penalty": cql_penalty,
            "current_q": current_q.mean(),
        }

    def update_target_networks(self, tau: float = 0.005) -> None:
        """Soft update of target networks.

        Args:
            tau: Soft update coefficient.
        """
        for target_param, param in zip(
            self.target_q_ensemble.parameters(),
            self.q_ensemble.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
