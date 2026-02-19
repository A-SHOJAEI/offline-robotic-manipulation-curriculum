#!/usr/bin/env python
"""Training script for curriculum learning."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from offline_robotic_manipulation_curriculum.training.trainer import CurriculumTrainer
from offline_robotic_manipulation_curriculum.utils.config import (
    load_config,
    validate_config,
    get_device,
)
from offline_robotic_manipulation_curriculum.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train offline RL curriculum learning model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging — configure root logger so all module loggers inherit handlers
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(
        name="train",
        log_file="logs/train.log",
        level=log_level,
    )
    # Also configure root logger for module-level loggers
    root = logging.getLogger()
    root.setLevel(log_level)
    if not root.handlers:
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(log_level)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    logger = logging.getLogger("train")

    logger.info("=" * 80)
    logger.info("Starting Offline Robotic Manipulation Curriculum Training")
    logger.info("=" * 80)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Override seed if provided
        if args.seed is not None:
            config["system"]["seed"] = args.seed
            logger.info(f"Overriding seed to {args.seed}")

        # Validate configuration
        validate_config(config)

        # Set random seeds
        seed = config["system"]["seed"]
        set_seeds(seed)
        logger.info(f"Set random seed to {seed}")

        # Log device information
        device = get_device(config)
        if device == "cuda":
            try:
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            except Exception as e:
                logger.warning(f"Could not get GPU information: {e}")

        # Create checkpoint directory
        checkpoint_dir = Path(config["system"]["checkpointing"]["save_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize trainer
        logger.info("Initializing curriculum trainer")
        trainer = CurriculumTrainer(config)

        # Resume from checkpoint if provided
        if args.checkpoint is not None:
            try:
                logger.info(f"Loading checkpoint from {args.checkpoint}")
                checkpoint = torch.load(args.checkpoint, map_location=device)
                trainer.agent.load_state_dict(checkpoint["agent_state_dict"])
                trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                trainer.global_epoch = checkpoint["epoch"]
                logger.info(f"Resumed from epoch {trainer.global_epoch}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                raise

        # Start training
        logger.info("Starting training loop")
        trainer.train()

        # Save final model
        try:
            final_model_path = models_dir / "final_model.pt"
            torch.save(trainer.agent.state_dict(), final_model_path)
            logger.info(f"Saved final model to {final_model_path}")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")

        # Copy best model to models directory
        try:
            best_checkpoint = checkpoint_dir / "best_model.pt"
            if best_checkpoint.exists():
                best_model_path = models_dir / "best_model.pt"
                checkpoint = torch.load(best_checkpoint, map_location=device)
                torch.save(checkpoint["agent_state_dict"], best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
