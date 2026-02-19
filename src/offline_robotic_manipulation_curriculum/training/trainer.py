"""Curriculum trainer for progressive offline RL."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from offline_robotic_manipulation_curriculum.data.loader import D4RLDataLoader
from offline_robotic_manipulation_curriculum.evaluation.metrics import evaluate_policy
from offline_robotic_manipulation_curriculum.models.model import (
    BehavioralCloningAgent,
    CQLAgent,
)
from offline_robotic_manipulation_curriculum.utils.logger import MetricsLogger

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """Scheduler for curriculum stage progression."""

    def __init__(
        self,
        stages: List[Dict],
        mode: str = "uncertainty_based",
        uncertainty_threshold: float = 0.15,
        min_stage_epochs: int = 10,
        max_stage_epochs: int = 50,
        progression_patience: int = 5,
    ):
        """Initialize curriculum scheduler.

        Args:
            stages: List of curriculum stage configurations.
            mode: Scheduling mode ('uncertainty_based', 'fixed_schedule', 'performance_based').
            uncertainty_threshold: Threshold for uncertainty-based progression.
            min_stage_epochs: Minimum epochs per stage.
            max_stage_epochs: Maximum epochs per stage.
            progression_patience: Patience for performance-based progression.
        """
        self.stages = sorted(stages, key=lambda x: x["difficulty"])
        self.mode = mode
        self.uncertainty_threshold = uncertainty_threshold
        self.min_stage_epochs = min_stage_epochs
        self.max_stage_epochs = max_stage_epochs
        self.progression_patience = progression_patience

        self.current_stage_idx = 0
        self.stage_epochs = 0
        self.best_stage_metric = -float("inf")
        self.patience_counter = 0

        logger.info(f"Initialized curriculum with {len(stages)} stages")

    def should_progress(
        self,
        epoch: int,
        metrics: Dict[str, float],
    ) -> bool:
        """Determine if curriculum should progress to next stage.

        Args:
            epoch: Current epoch.
            metrics: Dictionary of current metrics.

        Returns:
            True if should progress to next stage.
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False

        self.stage_epochs += 1

        # Minimum epochs requirement
        if self.stage_epochs < self.min_stage_epochs:
            return False

        # Maximum epochs limit
        if self.stage_epochs >= self.max_stage_epochs:
            logger.info(f"Reached max epochs for stage {self.current_stage_idx}")
            return True

        # Mode-specific progression logic
        if self.mode == "uncertainty_based":
            uncertainty = metrics.get("policy_uncertainty", float("inf"))
            if uncertainty < self.uncertainty_threshold:
                logger.info(
                    f"Uncertainty {uncertainty:.4f} below threshold {self.uncertainty_threshold}"
                )
                return True

        elif self.mode == "performance_based":
            current_metric = metrics.get("normalized_return", -float("inf"))
            threshold = self.stages[self.current_stage_idx]["success_threshold"]

            if current_metric > self.best_stage_metric:
                self.best_stage_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if current_metric >= threshold:
                logger.info(f"Reached performance threshold {threshold}")
                return True

            if self.patience_counter >= self.progression_patience:
                logger.info(f"Exceeded patience, progressing anyway")
                return True

        elif self.mode == "fixed_schedule":
            schedule_length = self.max_stage_epochs
            if self.stage_epochs >= schedule_length:
                return True

        return False

    def progress_stage(self) -> Dict:
        """Progress to next curriculum stage.

        Returns:
            Configuration for new stage.
        """
        self.current_stage_idx += 1
        self.stage_epochs = 0
        self.best_stage_metric = -float("inf")
        self.patience_counter = 0

        stage = self.get_current_stage()
        logger.info(f"Progressed to stage {self.current_stage_idx}: {stage['name']}")
        return stage

    def get_current_stage(self) -> Dict:
        """Get current curriculum stage configuration.

        Returns:
            Current stage dictionary.
        """
        return self.stages[self.current_stage_idx]


class CurriculumTrainer:
    """Trainer for curriculum-based offline RL."""

    def __init__(self, config: Dict):
        """Initialize curriculum trainer.

        Args:
            config: Configuration dictionary.
        """
        self.config = config

        # Set random seeds
        seed = config["system"]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config["system"]["use_gpu"] else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # Initialize curriculum scheduler
        self.scheduler = CurriculumScheduler(
            stages=config["curriculum"]["stages"],
            mode=config["curriculum"]["scheduling"]["mode"],
            uncertainty_threshold=config["curriculum"]["scheduling"]["uncertainty_threshold"],
            min_stage_epochs=config["curriculum"]["scheduling"]["min_stage_epochs"],
            max_stage_epochs=config["curriculum"]["scheduling"]["max_stage_epochs"],
            progression_patience=config["curriculum"]["scheduling"]["progression_patience"],
        )

        # Data loader
        self.data_loader = D4RLDataLoader(
            cache_dir=config["data"]["cache_dir"]
        )

        # Initialize logger
        self.metrics_logger = MetricsLogger(
            log_dir=config["system"]["logging"]["log_dir"],
            use_mlflow=config["system"]["logging"]["use_mlflow"],
            use_tensorboard=config["system"]["logging"]["use_tensorboard"],
        )

        # Model placeholders
        self.agent = None
        self.bc_agent = None
        self.optimizer = None
        self.current_env = None
        self.current_buffer = None

        # Training state
        self.global_epoch = 0
        self.best_return = -float("inf")

        # Setup checkpoint directory
        self.checkpoint_dir = Path(config["system"]["checkpointing"]["save_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def initialize_stage(self, stage_config: Dict) -> None:
        """Initialize models and data for a curriculum stage.

        Args:
            stage_config: Stage configuration dictionary.
        """
        env_name = stage_config["env_name"]
        logger.info(f"Initializing stage: {stage_config['name']} ({env_name})")

        # Load dataset
        self.current_buffer, self.current_env = self.data_loader.load_dataset(
            env_name=env_name,
            normalize_obs=self.config["data"]["normalize_observations"],
            normalize_rewards=self.config["data"]["normalize_rewards"],
            reward_scale=self.config["data"]["reward_scale"],
        )

        # Get environment dimensions
        obs_dim = self.current_env.observation_space.shape[0]
        act_dim = self.current_env.action_space.shape[0]

        # Check if agent needs (re)initialization due to dimension change
        needs_init = self.agent is None
        if self.agent is not None:
            prev_obs = self.agent.obs_dim
            prev_act = self.agent.act_dim
            if prev_obs != obs_dim or prev_act != act_dim:
                logger.info(f"Dimension change ({prev_obs},{prev_act}) -> ({obs_dim},{act_dim}), reinitializing agent")
                needs_init = True

        if needs_init:
            self.agent = CQLAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dims=self.config["model"]["hidden_dims"],
                activation=self.config["model"]["activation"],
                dropout=self.config["model"]["dropout"],
                ensemble_size=self.config["model"]["ensemble_size"],
                cql_alpha=self.config["model"]["cql"]["alpha"],
                min_q_weight=self.config["model"]["cql"]["min_q_weight"],
                use_lagrange=self.config["model"]["cql"]["use_lagrange"],
                lagrange_threshold=self.config["model"]["cql"]["lagrange_threshold"],
            ).to(self.device)

            self.optimizer = optim.Adam(
                self.agent.parameters(),
                lr=self.config["training"]["learning_rate"],
                weight_decay=self.config["training"]["optimizer"]["weight_decay"],
            )

        # Behavioral cloning warm-start
        if self.config["model"]["bc"]["enabled"]:
            self.bc_warmstart(obs_dim, act_dim)

    def bc_warmstart(self, obs_dim: int, act_dim: int) -> None:
        """Warm-start policy with behavioral cloning.

        Args:
            obs_dim: Observation dimension.
            act_dim: Action dimension.
        """
        logger.info("Starting behavioral cloning warm-start")

        bc_agent = BehavioralCloningAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=self.config["model"]["hidden_dims"],
            activation=self.config["model"]["activation"],
            dropout=self.config["model"]["dropout"],
        ).to(self.device)

        bc_optimizer = optim.Adam(
            bc_agent.parameters(),
            lr=self.config["model"]["bc"]["learning_rate"],
        )

        bc_epochs = self.config["model"]["bc"]["warmstart_epochs"]
        batch_size = self.config["training"]["batch_size"]

        for epoch in range(bc_epochs):
            batch = self.current_buffer.sample(batch_size)
            obs = batch["observations"]
            actions = batch["actions"]

            loss = bc_agent.compute_loss(obs, actions)

            bc_optimizer.zero_grad()
            loss.backward()
            bc_optimizer.step()

            if (epoch + 1) % 5 == 0:
                logger.info(f"BC Epoch {epoch + 1}/{bc_epochs}, Loss: {loss.item():.4f}")

        # Transfer weights to CQL policy
        self.agent.policy.load_state_dict(bc_agent.policy.state_dict())
        logger.info("Completed BC warm-start, transferred weights to CQL policy")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics.
        """
        batch_size = self.config["training"]["batch_size"]
        gamma = self.config["training"]["gamma"]

        # Sample batch
        batch = self.current_buffer.sample(batch_size)

        # Compute CQL loss
        loss_dict = self.agent.compute_cql_loss(
            obs=batch["observations"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            next_obs=batch["next_observations"],
            terminals=batch["terminals"],
            gamma=gamma,
        )

        # Update agent
        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target networks
        if self.global_epoch % self.config["training"]["target_update_freq"] == 0:
            self.agent.update_target_networks(tau=self.config["training"]["tau"])

        # Convert to float for logging
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

        return metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate current policy.

        Returns:
            Dictionary of evaluation metrics.
        """
        eval_metrics = evaluate_policy(
            agent=self.agent,
            env=self.current_env,
            num_episodes=self.config["evaluation"]["eval_episodes"],
            deterministic=self.config["evaluation"]["deterministic"],
            device=self.device,
        )

        # Compute epistemic uncertainty
        batch = self.current_buffer.sample(min(1000, len(self.current_buffer)))
        with torch.no_grad():
            actions = self.agent.select_action(
                batch["observations"].cpu().numpy(),
                deterministic=True,
            )
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            uncertainty = self.agent.q_ensemble.get_uncertainty(
                batch["observations"],
                actions_tensor,
            )
            eval_metrics["policy_uncertainty"] = uncertainty.mean().item()

        return eval_metrics

    def save_checkpoint(self, filename: str = "checkpoint.pt") -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.global_epoch,
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state": {
                "current_stage_idx": self.scheduler.current_stage_idx,
                "stage_epochs": self.scheduler.stage_epochs,
            },
            "config": self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self) -> None:
        """Execute curriculum training."""
        logger.info("Starting curriculum training")

        try:
            # Initialize MLflow
            if self.config["system"]["logging"]["use_mlflow"]:
                try:
                    import mlflow
                    mlflow.set_experiment(self.config["system"]["logging"]["experiment_name"])
                    mlflow.start_run(run_name=self.config["system"]["logging"].get("run_name"))
                    mlflow.log_params({"seed": self.config["system"]["seed"]})
                except Exception as e:
                    logger.warning(f"MLflow initialization failed: {e}")

            # Main training loop
            while self.scheduler.current_stage_idx < len(self.scheduler.stages):
                # Initialize current stage
                stage_config = self.scheduler.get_current_stage()
                logger.info(f"=== Stage {self.scheduler.current_stage_idx + 1}/{len(self.scheduler.stages)}: {stage_config['name']} ===")
                self.initialize_stage(stage_config)

                # Train on current stage
                stage_start_epoch = self.global_epoch

                for epoch_in_stage in range(self.scheduler.max_stage_epochs):
                    # Training epoch
                    train_metrics = self.train_epoch()
                    self.global_epoch += 1

                    # Always print epoch progress
                    loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
                    print(f"[Stage {self.scheduler.current_stage_idx + 1}] Epoch {epoch_in_stage + 1}/{self.scheduler.max_stage_epochs} (global {self.global_epoch}) - {loss_str}", flush=True)

                    # Logging
                    if self.global_epoch % self.config["system"]["logging"]["log_freq"] == 0:
                        self.metrics_logger.log_metrics(
                            train_metrics,
                            step=self.global_epoch,
                            prefix="train/",
                        )

                    # Evaluation
                    if self.global_epoch % self.config["evaluation"]["eval_freq"] == 0:
                        eval_metrics = self.evaluate()
                        self.metrics_logger.log_metrics(
                            eval_metrics,
                            step=self.global_epoch,
                            prefix="eval/",
                        )

                        # Save best model
                        if eval_metrics["normalized_return"] > self.best_return:
                            self.best_return = eval_metrics["normalized_return"]
                            self.save_checkpoint("best_model.pt")

                        # Check curriculum progression
                        if self.scheduler.should_progress(self.global_epoch, eval_metrics):
                            break

                    # Periodic checkpoint
                    if self.global_epoch % self.config["system"]["checkpointing"]["save_freq"] == 0:
                        self.save_checkpoint(f"checkpoint_epoch_{self.global_epoch}.pt")

                # Progress to next stage if not on last stage
                if self.scheduler.current_stage_idx < len(self.scheduler.stages) - 1:
                    self.scheduler.progress_stage()
                else:
                    logger.info("Final curriculum stage completed")
                    break

            logger.info("Training completed successfully")

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        finally:
            # Cleanup
            if self.config["system"]["logging"]["use_mlflow"]:
                try:
                    import mlflow
                    mlflow.end_run()
                except Exception:
                    pass

            self.metrics_logger.close()
