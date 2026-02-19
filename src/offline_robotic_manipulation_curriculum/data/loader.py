"""Data loading utilities for D4RL datasets."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset
from gymnasium import spaces

logger = logging.getLogger(__name__)


class MockEnv:
    """Mock environment for demo mode when d4rl is not available."""

    def __init__(self, obs_dim: int, act_dim: int, max_steps: int = 200):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        self.max_steps = max_steps
        self._step_count = 0
        self.spec = type("Spec", (), {"id": "mock-env-v0"})()

    def reset(self, **kwargs):
        self._step_count = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._step_count += 1
        obs = self.observation_space.sample()
        reward = float(np.random.normal(0.0, 0.1))
        terminated = np.random.random() < 0.01
        truncated = self._step_count >= self.max_steps
        info = {"success": np.random.random() < 0.05}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


class ReplayBuffer(Dataset):
    """Replay buffer for offline RL datasets."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
        device: str = "cpu",
    ):
        """Initialize replay buffer.

        Args:
            observations: Array of observations [N, obs_dim].
            actions: Array of actions [N, act_dim].
            rewards: Array of rewards [N].
            next_observations: Array of next observations [N, obs_dim].
            terminals: Array of terminal flags [N].
            device: Device to store tensors on.
        """
        self.observations = torch.FloatTensor(observations).to(device)
        self.actions = torch.FloatTensor(actions).to(device)
        self.rewards = torch.FloatTensor(rewards).to(device)
        self.next_observations = torch.FloatTensor(next_observations).to(device)
        self.terminals = torch.FloatTensor(terminals).to(device)

        self.size = len(observations)
        self.device = device

        logger.info(f"Initialized replay buffer with {self.size} transitions")

    def __len__(self) -> int:
        """Return buffer size."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get transition at index.

        Args:
            idx: Index of transition.

        Returns:
            Tuple of (observation, action, reward, next_observation, terminal).
        """
        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_observations[idx],
            self.terminals[idx],
        )

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch from buffer.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary containing batch of transitions.
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "terminals": self.terminals[indices],
        }


class D4RLDataLoader:
    """Data loader for D4RL datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize D4RL data loader.

        Args:
            cache_dir: Optional directory for caching datasets.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_synthetic_dataset(
        self,
        env_name: str,
        num_transitions: int = 10000
    ) -> Dict[str, np.ndarray]:
        """Generate synthetic dataset for demo purposes.

        Args:
            env_name: Name of environment (determines dimensions).
            num_transitions: Number of transitions to generate.

        Returns:
            Dictionary with dataset components.
        """
        logger.warning(
            f"d4rl not available - generating synthetic data for {env_name}. "
            "Install d4rl for real datasets: pip install git+https://github.com/Farama-Foundation/D4RL.git"
        )

        # Environment-specific dimensions
        env_dims = {
            "pen-human-v1": (45, 24),
            "door-human-v1": (39, 28),
            "kitchen-partial-v0": (60, 9),
            "antmaze-large-diverse-v2": (29, 8),
        }

        obs_dim, act_dim = env_dims.get(env_name, (20, 4))

        # Generate synthetic trajectories
        observations = np.random.randn(num_transitions, obs_dim).astype(np.float32)
        actions = np.random.randn(num_transitions, act_dim).astype(np.float32)

        # Generate plausible rewards (with some structure)
        rewards = np.random.randn(num_transitions).astype(np.float32) * 0.5 + 0.1

        # Simple dynamics: next_obs = obs + small perturbation
        next_observations = observations + np.random.randn(num_transitions, obs_dim).astype(np.float32) * 0.1

        # Terminal flags (episode boundaries every ~200 steps)
        terminals = np.zeros(num_transitions, dtype=np.float32)
        terminals[::200] = 1.0

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "terminals": terminals,
        }

    def load_dataset(
        self,
        env_name: str,
        normalize_obs: bool = True,
        normalize_rewards: bool = False,
        reward_scale: float = 1.0,
    ) -> Tuple[ReplayBuffer, gym.Env]:
        """Load D4RL dataset and create replay buffer.

        Args:
            env_name: Name of D4RL environment.
            normalize_obs: Whether to normalize observations.
            normalize_rewards: Whether to normalize rewards.
            reward_scale: Scale factor for rewards.

        Returns:
            Tuple of (replay_buffer, environment).

        Raises:
            RuntimeError: If dataset cannot be loaded.
        """
        try:
            import d4rl
            d4rl_available = True
        except ImportError:
            d4rl_available = False
            logger.warning("d4rl not installed - using synthetic demo data")

        logger.info(f"Loading dataset: {env_name}")

        if d4rl_available:
            try:
                env = gym.make(env_name)
                dataset = d4rl.qlearning_dataset(env)
            except Exception as e:
                logger.warning(f"Failed to load d4rl dataset {env_name}: {e}. Using synthetic data.")
                dataset = self._generate_synthetic_dataset(env_name)
                # Create mock environment with appropriate dimensions
                env_dims = {
                    "pen-human-v1": (45, 24),
                    "door-human-v1": (39, 28),
                    "kitchen-partial-v0": (60, 9),
                    "antmaze-large-diverse-v2": (29, 8),
                }
                obs_dim, act_dim = env_dims.get(env_name, (20, 4))
                env = MockEnv(obs_dim, act_dim)
        else:
            dataset = self._generate_synthetic_dataset(env_name)
            # Create mock environment with appropriate dimensions
            env_dims = {
                "pen-human-v1": (45, 24),
                "door-human-v1": (39, 28),
                "kitchen-partial-v0": (60, 9),
                "antmaze-large-diverse-v2": (29, 8),
            }
            obs_dim, act_dim = env_dims.get(env_name, (20, 4))
            env = MockEnv(obs_dim, act_dim)

        # Extract components
        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        next_observations = dataset["next_observations"]
        terminals = dataset["terminals"]

        # Apply preprocessing
        if normalize_obs:
            obs_mean = observations.mean(axis=0)
            obs_std = observations.std(axis=0) + 1e-8
            observations = (observations - obs_mean) / obs_std
            next_observations = (next_observations - obs_mean) / obs_std
            logger.info("Normalized observations")

        if normalize_rewards:
            reward_mean = rewards.mean()
            reward_std = rewards.std() + 1e-8
            rewards = (rewards - reward_mean) / reward_std
            logger.info("Normalized rewards")

        if reward_scale != 1.0:
            rewards = rewards * reward_scale
            logger.info(f"Scaled rewards by {reward_scale}")

        # Create replay buffer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        replay_buffer = ReplayBuffer(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            device=device,
        )

        logger.info(
            f"Loaded {len(replay_buffer)} transitions from {env_name} "
            f"(obs_dim={observations.shape[1]}, act_dim={actions.shape[1]})"
        )

        return replay_buffer, env

    def get_env_info(self, env_name: str) -> Dict[str, int]:
        """Get environment dimensions.

        Args:
            env_name: Name of D4RL environment.

        Returns:
            Dictionary with observation and action dimensions.
        """
        try:
            import d4rl
            env = gym.make(env_name)

            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]

            env.close()

            return {"obs_dim": obs_dim, "act_dim": act_dim}
        except Exception as e:
            logger.warning(f"Failed to get env info from d4rl for {env_name}: {e}. Using defaults.")

            # Return default dimensions based on environment name
            env_dims = {
                "pen-human-v1": (45, 24),
                "door-human-v1": (39, 28),
                "kitchen-partial-v0": (60, 9),
                "antmaze-large-diverse-v2": (29, 8),
            }

            obs_dim, act_dim = env_dims.get(env_name, (20, 4))
            return {"obs_dim": obs_dim, "act_dim": act_dim}
