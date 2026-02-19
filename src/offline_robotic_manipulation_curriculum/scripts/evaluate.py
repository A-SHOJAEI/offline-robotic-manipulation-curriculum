#!/usr/bin/env python
"""Evaluation script for trained policies."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
import torch

from offline_robotic_manipulation_curriculum.models.model import CQLAgent
from offline_robotic_manipulation_curriculum.evaluation.metrics import (
    evaluate_policy,
    compute_normalized_return,
    compute_policy_robustness,
)
from offline_robotic_manipulation_curriculum.utils.config import load_config
from offline_robotic_manipulation_curriculum.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (overrides config)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Evaluate policy robustness",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.txt",
        help="Output file for results",
    )
    return parser.parse_args()


def load_agent(checkpoint_path: str, config: Dict, device: str) -> CQLAgent:
    """Load trained agent from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Loaded CQL agent.

    Raises:
        FileNotFoundError: If checkpoint file not found.
        KeyError: If checkpoint has invalid format.
        RuntimeError: If model loading fails.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Get model architecture parameters
    model_config = config["model"]

    # Initialize agent (dimensions will be set from checkpoint)
    try:
        agent = CQLAgent(
            obs_dim=checkpoint["agent_state_dict"]["policy.backbone.0.weight"].shape[1],
            act_dim=checkpoint["agent_state_dict"]["policy.mean_head.weight"].shape[0],
            hidden_dims=model_config["hidden_dims"],
            activation=model_config["activation"],
            dropout=model_config["dropout"],
            ensemble_size=model_config["ensemble_size"],
            cql_alpha=model_config["cql"]["alpha"],
            min_q_weight=model_config["cql"]["min_q_weight"],
            use_lagrange=model_config["cql"]["use_lagrange"],
            lagrange_threshold=model_config["cql"]["lagrange_threshold"],
        ).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize agent: {e}")

    # Load weights
    try:
        if "agent_state_dict" in checkpoint:
            agent.load_state_dict(checkpoint["agent_state_dict"])
        else:
            agent.load_state_dict(checkpoint)
    except Exception as e:
        raise RuntimeError(f"Failed to load agent state dict: {e}")

    agent.eval()

    return agent


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    logger = setup_logger(
        name="evaluate",
        log_file="logs/evaluate.log",
        level=logging.INFO,
    )

    logger.info("=" * 80)
    logger.info("Starting Policy Evaluation")
    logger.info("=" * 80)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load agent
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        try:
            agent = load_agent(args.checkpoint, config, device)
            logger.info("Agent loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            raise

        # Determine environment
        if args.env is not None:
            env_name = args.env
        else:
            # Use last stage from curriculum
            env_name = config["curriculum"]["stages"][-1]["env_name"]

        logger.info(f"Evaluating on environment: {env_name}")

        # Create environment
        try:
            import d4rl
            env = gym.make(env_name)
        except ImportError:
            logger.error("D4RL not installed. Install with: pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            sys.exit(1)

        # Evaluate policy
        logger.info(f"Running evaluation for {args.num_episodes} episodes")
        try:
            metrics = evaluate_policy(
                agent=agent,
                env=env,
                num_episodes=args.num_episodes,
                deterministic=args.deterministic,
                device=device,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

        # Log results
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Results")
        logger.info("=" * 80)
        logger.info(f"Environment: {env_name}")
        logger.info(f"Episodes: {args.num_episodes}")
        logger.info(f"Average Return: {metrics['average_return']:.2f} ± {metrics['std_return']:.2f}")
        logger.info(f"Min/Max Return: {metrics['min_return']:.2f} / {metrics['max_return']:.2f}")
        logger.info(f"Normalized Return: {metrics['normalized_return']:.4f}")
        logger.info(f"Success Rate: {metrics['success_rate']:.2%}")
        logger.info(f"Average Length: {metrics['average_length']:.1f}")

        # Robustness evaluation
        if args.robustness:
            logger.info("\nEvaluating policy robustness...")
            try:
                robustness_metrics = compute_policy_robustness(
                    agent=agent,
                    env=env,
                    num_episodes=min(20, args.num_episodes),
                    noise_levels=[0.0, 0.05, 0.1, 0.2],
                )
                logger.info(f"Robustness Score: {robustness_metrics['robustness_score']:.4f}")
                logger.info("Returns by noise level:")
                for noise, ret in robustness_metrics['returns_by_noise'].items():
                    logger.info(f"  Noise {noise:.2f}: {ret:.2f}")
            except Exception as e:
                logger.error(f"Robustness evaluation failed: {e}")

        # Save results to file
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                f.write("Evaluation Results\n")
                f.write("=" * 80 + "\n")
                f.write(f"Environment: {env_name}\n")
                f.write(f"Checkpoint: {args.checkpoint}\n")
                f.write(f"Episodes: {args.num_episodes}\n")
                f.write(f"\nMetrics:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")

                if args.robustness:
                    f.write(f"\nRobustness Score: {robustness_metrics['robustness_score']:.4f}\n")

            logger.info(f"\nResults saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
