"""Data preprocessing utilities for offline RL."""

import logging
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def normalize_observations(
    observations: np.ndarray,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize observations using z-score normalization.

    Args:
        observations: Array of observations [N, obs_dim].
        mean: Optional pre-computed mean. If None, computed from data.
        std: Optional pre-computed std. If None, computed from data.
        eps: Small constant for numerical stability.

    Returns:
        Tuple of (normalized_observations, mean, std).
    """
    if mean is None:
        mean = observations.mean(axis=0)
    if std is None:
        std = observations.std(axis=0) + eps

    normalized = (observations - mean) / std

    return normalized, mean, std


def normalize_rewards(
    rewards: np.ndarray,
    method: str = "standardize",
    clip_range: Tuple[float, float] = None,
) -> np.ndarray:
    """Normalize rewards.

    Args:
        rewards: Array of rewards [N].
        method: Normalization method ('standardize', 'min_max', or 'none').
        clip_range: Optional tuple for clipping rewards.

    Returns:
        Normalized rewards.
    """
    if method == "standardize":
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        normalized = (rewards - mean) / std
    elif method == "min_max":
        min_val = rewards.min()
        max_val = rewards.max()
        normalized = (rewards - min_val) / (max_val - min_val + 1e-8)
    elif method == "none":
        normalized = rewards
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if clip_range is not None:
        normalized = np.clip(normalized, clip_range[0], clip_range[1])

    return normalized


def compute_returns(
    rewards: np.ndarray,
    terminals: np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    """Compute discounted returns for each trajectory.

    Args:
        rewards: Array of rewards [N].
        terminals: Array of terminal flags [N].
        gamma: Discount factor.

    Returns:
        Array of returns [N].
    """
    returns = np.zeros_like(rewards)
    running_return = 0.0

    # Compute returns backwards through trajectories
    for t in reversed(range(len(rewards))):
        if terminals[t]:
            running_return = 0.0
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def compute_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminals: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> np.ndarray:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Array of rewards [N].
        values: Array of state values [N].
        next_values: Array of next state values [N].
        terminals: Array of terminal flags [N].
        gamma: Discount factor.
        lam: GAE lambda parameter.

    Returns:
        Array of advantages [N].
    """
    advantages = np.zeros_like(rewards)
    last_advantage = 0.0

    for t in reversed(range(len(rewards))):
        if terminals[t]:
            last_advantage = 0.0

        delta = rewards[t] + gamma * next_values[t] * (1 - terminals[t]) - values[t]
        last_advantage = delta + gamma * lam * (1 - terminals[t]) * last_advantage
        advantages[t] = last_advantage

    return advantages


def augment_transitions(
    observations: np.ndarray,
    actions: np.ndarray,
    noise_scale: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply data augmentation to transitions.

    Args:
        observations: Array of observations [N, obs_dim].
        actions: Array of actions [N, act_dim].
        noise_scale: Scale of Gaussian noise to add.

    Returns:
        Tuple of (augmented_observations, augmented_actions).
    """
    obs_noise = np.random.normal(0, noise_scale, observations.shape)
    act_noise = np.random.normal(0, noise_scale, actions.shape)

    augmented_obs = observations + obs_noise
    augmented_act = actions + act_noise

    return augmented_obs, augmented_act


def filter_trajectories_by_return(
    dataset: dict,
    percentile: float = 50.0,
) -> dict:
    """Filter dataset to keep only high-return trajectories.

    Args:
        dataset: Dictionary containing dataset arrays.
        percentile: Percentile threshold for filtering.

    Returns:
        Filtered dataset dictionary.
    """
    rewards = dataset["rewards"]
    terminals = dataset["terminals"]

    # Compute trajectory returns
    traj_returns = []
    current_return = 0.0

    for r, done in zip(rewards, terminals):
        current_return += r
        if done:
            traj_returns.append(current_return)
            current_return = 0.0

    # Compute threshold
    threshold = np.percentile(traj_returns, percentile)
    logger.info(f"Filtering trajectories with return >= {threshold:.2f}")

    # Filter transitions
    filtered_indices = []
    current_return = 0.0
    traj_start_idx = 0

    for i, (r, done) in enumerate(zip(rewards, terminals)):
        current_return += r
        if done:
            if current_return >= threshold:
                filtered_indices.extend(range(traj_start_idx, i + 1))
            current_return = 0.0
            traj_start_idx = i + 1

    # Create filtered dataset
    filtered_dataset = {
        key: val[filtered_indices] for key, val in dataset.items()
    }

    logger.info(
        f"Filtered dataset from {len(rewards)} to {len(filtered_indices)} transitions"
    )

    return filtered_dataset
