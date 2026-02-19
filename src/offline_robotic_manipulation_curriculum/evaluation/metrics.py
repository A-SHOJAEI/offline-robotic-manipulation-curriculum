"""Evaluation metrics for offline RL policies."""

import logging
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger(__name__)


def evaluate_policy(
    agent: torch.nn.Module,
    env: gym.Env,
    num_episodes: int = 50,
    deterministic: bool = True,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate policy performance in environment.

    Args:
        agent: Policy agent to evaluate.
        env: Gymnasium environment.
        num_episodes: Number of evaluation episodes.
        deterministic: Whether to use deterministic actions.
        device: Device for computation.
        max_episode_steps: Maximum steps per episode.

    Returns:
        Dictionary of evaluation metrics.
    """
    agent.eval()

    episode_returns = []
    episode_lengths = []
    success_count = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

            # Check for max steps
            if max_episode_steps is not None and episode_length >= max_episode_steps:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # Check success (environment-specific)
        if info.get("success", False):
            success_count += 1

    agent.train()

    # Compute metrics
    metrics = {
        "average_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "min_return": np.min(episode_returns),
        "max_return": np.max(episode_returns),
        "average_length": np.mean(episode_lengths),
        "success_rate": success_count / num_episodes,
    }

    # Compute normalized return if possible
    try:
        normalized_return = compute_normalized_return(
            returns=episode_returns,
            env_name=env.spec.id if hasattr(env, "spec") else "unknown",
        )
        metrics["normalized_return"] = normalized_return
    except Exception as e:
        logger.warning(f"Could not compute normalized return: {e}")
        metrics["normalized_return"] = metrics["average_return"]

    logger.info(
        f"Evaluation: Return={metrics['average_return']:.2f} "
        f"(±{metrics['std_return']:.2f}), "
        f"Success={metrics['success_rate']:.2%}"
    )

    return metrics


def compute_normalized_return(
    returns: list,
    env_name: str,
) -> float:
    """Compute normalized return using D4RL reference scores.

    Args:
        returns: List of episode returns.
        env_name: Name of environment.

    Returns:
        Normalized return in [0, 1].
    """
    # D4RL reference scores (approximate)
    reference_scores = {
        "pen-human-v1": {"min": -3.0, "max": 50.0},
        "door-human-v1": {"min": -0.1, "max": 350.0},
        "kitchen-partial-v0": {"min": 0.0, "max": 4.0},
        "antmaze-large-diverse-v2": {"min": 0.0, "max": 1.0},
    }

    mean_return = np.mean(returns)

    # Get reference scores for environment
    if env_name in reference_scores:
        ref = reference_scores[env_name]
        normalized = (mean_return - ref["min"]) / (ref["max"] - ref["min"])
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        # Fallback: use raw return
        normalized = mean_return
        logger.warning(f"No reference scores for {env_name}, using raw return")

    return normalized


def compute_success_rate(
    agent: torch.nn.Module,
    env: gym.Env,
    num_episodes: int = 50,
    success_threshold: float = 1.0,
) -> float:
    """Compute success rate for goal-conditioned tasks.

    Args:
        agent: Policy agent.
        env: Gymnasium environment.
        num_episodes: Number of episodes.
        success_threshold: Threshold for considering episode successful.

    Returns:
        Success rate in [0, 1].
    """
    agent.eval()
    success_count = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward

        # Check success criteria
        if info.get("success", False) or episode_return >= success_threshold:
            success_count += 1

    agent.train()

    return success_count / num_episodes


def compute_policy_robustness(
    agent: torch.nn.Module,
    env: gym.Env,
    num_episodes: int = 50,
    noise_levels: list = [0.0, 0.05, 0.1, 0.2],
) -> Dict[str, float]:
    """Compute policy robustness to observation noise.

    Args:
        agent: Policy agent.
        env: Gymnasium environment.
        num_episodes: Number of episodes per noise level.
        noise_levels: List of noise standard deviations.

    Returns:
        Dictionary of robustness metrics.
    """
    agent.eval()

    returns_by_noise = {}

    for noise_std in noise_levels:
        episode_returns = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_return = 0.0

            while not done:
                # Add observation noise
                noisy_obs = obs + np.random.normal(0, noise_std, size=obs.shape)

                action = agent.select_action(noisy_obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward

            episode_returns.append(episode_return)

        returns_by_noise[noise_std] = np.mean(episode_returns)

    agent.train()

    # Compute robustness score
    baseline_return = returns_by_noise[0.0]
    robustness_scores = []

    for noise_std in noise_levels[1:]:
        if baseline_return != 0:
            score = returns_by_noise[noise_std] / baseline_return
        else:
            score = 0.0
        robustness_scores.append(score)

    metrics = {
        "robustness_score": np.mean(robustness_scores),
        "returns_by_noise": returns_by_noise,
    }

    return metrics


def compute_curriculum_transfer_efficiency(
    stage_returns: Dict[str, list],
    baseline_returns: Dict[str, list],
) -> float:
    """Compute curriculum transfer efficiency metric.

    Args:
        stage_returns: Dictionary mapping stage names to returns.
        baseline_returns: Dictionary mapping stage names to baseline returns.

    Returns:
        Transfer efficiency score.
    """
    efficiency_scores = []

    for stage_name in stage_returns:
        if stage_name in baseline_returns:
            curriculum_mean = np.mean(stage_returns[stage_name])
            baseline_mean = np.mean(baseline_returns[stage_name])

            if baseline_mean != 0:
                efficiency = curriculum_mean / baseline_mean
                efficiency_scores.append(efficiency)

    if len(efficiency_scores) == 0:
        return 0.0

    return np.mean(efficiency_scores)
