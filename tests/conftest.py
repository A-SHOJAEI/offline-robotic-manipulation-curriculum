"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture(scope="session")
def device():
    """Device fixture for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="function")
def random_seed():
    """Set random seed for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "curriculum": {
            "stages": [
                {
                    "name": "test-stage-1",
                    "env_name": "pen-human-v1",
                    "difficulty": 1,
                    "min_episodes": 100,
                    "success_threshold": 0.4,
                },
                {
                    "name": "test-stage-2",
                    "env_name": "door-human-v1",
                    "difficulty": 2,
                    "min_episodes": 100,
                    "success_threshold": 0.5,
                }
            ],
            "scheduling": {
                "mode": "uncertainty_based",
                "uncertainty_threshold": 0.15,
                "min_stage_epochs": 5,
                "max_stage_epochs": 20,
                "progression_patience": 3,
            },
        },
        "model": {
            "architecture": "cql",
            "hidden_dims": [128, 128],
            "activation": "relu",
            "dropout": 0.1,
            "ensemble_size": 3,
            "cql": {
                "alpha": 5.0,
                "min_q_weight": 5.0,
                "lagrange_threshold": 10.0,
                "use_lagrange": True,
            },
            "bc": {
                "enabled": True,
                "warmstart_epochs": 5,
                "learning_rate": 0.0003,
            },
        },
        "training": {
            "total_epochs": 50,
            "batch_size": 64,
            "buffer_size": 10000,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "tau": 0.005,
            "target_update_freq": 2,
            "optimizer": {
                "type": "adam",
                "weight_decay": 0.0001,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "type": "cosine",
                "warmup_epochs": 5,
                "min_lr": 0.00001,
            },
        },
        "evaluation": {
            "eval_episodes": 10,
            "eval_freq": 5,
            "deterministic": True,
            "save_video": False,
            "metrics": ["normalized_return", "success_rate"],
        },
        "data": {
            "cache_dir": "./test_cache",
            "preload_all": False,
            "normalize_observations": True,
            "normalize_rewards": False,
            "reward_scale": 1.0,
        },
        "system": {
            "seed": 42,
            "device": "auto",
            "num_workers": 2,
            "use_gpu": False,
            "mixed_precision": False,
            "checkpointing": {
                "save_dir": "./test_checkpoints",
                "save_freq": 10,
                "keep_last_n": 3,
                "save_best": True,
            },
            "logging": {
                "log_dir": "./test_logs",
                "log_freq": 1,
                "use_mlflow": False,
                "use_tensorboard": False,
                "experiment_name": "test_experiment",
                "run_name": "test_run",
            },
        },
    }


@pytest.fixture
def sample_batch(device):
    """Sample batch of transitions for testing."""
    batch_size = 32
    obs_dim = 24
    act_dim = 4

    return {
        "observations": torch.randn(batch_size, obs_dim).to(device),
        "actions": torch.randn(batch_size, act_dim).to(device),
        "rewards": torch.randn(batch_size).to(device),
        "next_observations": torch.randn(batch_size, obs_dim).to(device),
        "terminals": torch.zeros(batch_size).to(device),
    }


@pytest.fixture
def mock_replay_buffer(device):
    """Mock replay buffer for testing."""
    from offline_robotic_manipulation_curriculum.data.loader import ReplayBuffer

    size = 1000
    obs_dim = 24
    act_dim = 4

    observations = np.random.randn(size, obs_dim)
    actions = np.random.randn(size, act_dim)
    rewards = np.random.randn(size)
    next_observations = np.random.randn(size, obs_dim)
    terminals = np.random.randint(0, 2, size=size).astype(float)

    return ReplayBuffer(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        device=str(device),
    )
