"""Tests for training components."""

import pytest
import torch

from offline_robotic_manipulation_curriculum.training.trainer import (
    CurriculumScheduler,
    CurriculumTrainer,
)


class TestCurriculumScheduler:
    """Tests for curriculum scheduler."""

    def test_initialization(self, sample_config):
        """Test scheduler initialization."""
        stages = sample_config["curriculum"]["stages"]
        scheduling = sample_config["curriculum"]["scheduling"]

        scheduler = CurriculumScheduler(
            stages=stages,
            mode=scheduling["mode"],
            uncertainty_threshold=scheduling["uncertainty_threshold"],
            min_stage_epochs=scheduling["min_stage_epochs"],
            max_stage_epochs=scheduling["max_stage_epochs"],
            progression_patience=scheduling["progression_patience"],
        )

        assert scheduler.current_stage_idx == 0
        assert len(scheduler.stages) == len(stages)

    def test_get_current_stage(self, sample_config):
        """Test getting current stage."""
        stages = sample_config["curriculum"]["stages"]
        scheduling = sample_config["curriculum"]["scheduling"]

        scheduler = CurriculumScheduler(
            stages=stages,
            mode=scheduling["mode"],
        )

        current_stage = scheduler.get_current_stage()
        assert current_stage["name"] == stages[0]["name"]

    def test_min_epochs_requirement(self, sample_config):
        """Test minimum epochs requirement."""
        stages = sample_config["curriculum"]["stages"]
        scheduling = sample_config["curriculum"]["scheduling"]

        scheduler = CurriculumScheduler(
            stages=stages,
            mode="uncertainty_based",
            min_stage_epochs=10,
        )

        # Should not progress before minimum epochs
        for _ in range(9):
            should_progress = scheduler.should_progress(
                epoch=scheduler.stage_epochs,
                metrics={"policy_uncertainty": 0.05},
            )
            assert not should_progress

    def test_max_epochs_limit(self, sample_config):
        """Test maximum epochs limit."""
        stages = sample_config["curriculum"]["stages"]

        scheduler = CurriculumScheduler(
            stages=stages,
            mode="uncertainty_based",
            min_stage_epochs=5,
            max_stage_epochs=10,
        )

        # Should progress after maximum epochs
        # stage_epochs starts at 0, increments to 1, 2, ..., 9 in first 9 calls
        for i in range(9):
            should_progress = scheduler.should_progress(
                epoch=i,
                metrics={"policy_uncertainty": 0.5},
            )
            if i < 4:  # stage_epochs < 5
                assert not should_progress
            # else: still not at max

        # On the 10th call, stage_epochs becomes 10, which >= max_stage_epochs
        should_progress = scheduler.should_progress(
            epoch=10,
            metrics={"policy_uncertainty": 0.5},
        )
        assert should_progress

    def test_uncertainty_based_progression(self, sample_config):
        """Test uncertainty-based progression."""
        stages = sample_config["curriculum"]["stages"]

        scheduler = CurriculumScheduler(
            stages=stages,
            mode="uncertainty_based",
            uncertainty_threshold=0.15,
            min_stage_epochs=5,
            max_stage_epochs=50,
        )

        # Advance past minimum epochs (call should_progress 5 times)
        for i in range(5):
            scheduler.should_progress(epoch=i, metrics={"policy_uncertainty": 0.3})

        # High uncertainty - should not progress
        should_progress = scheduler.should_progress(
            epoch=6,
            metrics={"policy_uncertainty": 0.3},
        )
        assert not should_progress

        # Low uncertainty - should progress
        should_progress = scheduler.should_progress(
            epoch=7,
            metrics={"policy_uncertainty": 0.1},
        )
        assert should_progress

    def test_performance_based_progression(self, sample_config):
        """Test performance-based progression."""
        stages = sample_config["curriculum"]["stages"]

        scheduler = CurriculumScheduler(
            stages=stages,
            mode="performance_based",
            min_stage_epochs=5,
            max_stage_epochs=50,
        )

        # Advance past minimum epochs (call should_progress 5 times)
        for i in range(5):
            scheduler.should_progress(epoch=i, metrics={"normalized_return": 0.2})

        # Below threshold - should not progress
        should_progress = scheduler.should_progress(
            epoch=6,
            metrics={"normalized_return": 0.2},
        )
        assert not should_progress

        # Above threshold - should progress
        should_progress = scheduler.should_progress(
            epoch=7,
            metrics={"normalized_return": 0.5},
        )
        assert should_progress

    def test_progress_stage(self, sample_config):
        """Test stage progression."""
        stages = [
            {"name": "stage1", "difficulty": 1, "env_name": "env1", "success_threshold": 0.4, "min_episodes": 100},
            {"name": "stage2", "difficulty": 2, "env_name": "env2", "success_threshold": 0.5, "min_episodes": 100},
        ]

        scheduler = CurriculumScheduler(stages=stages, mode="fixed_schedule")

        initial_idx = scheduler.current_stage_idx
        new_stage = scheduler.progress_stage()

        assert scheduler.current_stage_idx == initial_idx + 1
        assert new_stage["name"] == "stage2"
        assert scheduler.stage_epochs == 0

    def test_fixed_schedule_mode(self, sample_config):
        """Test fixed schedule mode."""
        stages = sample_config["curriculum"]["stages"]

        scheduler = CurriculumScheduler(
            stages=stages,
            mode="fixed_schedule",
            min_stage_epochs=5,
            max_stage_epochs=10,
        )

        # Should progress after max epochs regardless of metrics
        # Call should_progress 9 times
        for i in range(9):
            should_progress = scheduler.should_progress(
                epoch=i,
                metrics={"normalized_return": 0.0},
            )
            assert not should_progress

        # On the 10th call, should progress
        should_progress = scheduler.should_progress(
            epoch=10,
            metrics={"normalized_return": 0.0},
        )
        assert should_progress


class TestCurriculumTrainer:
    """Tests for curriculum trainer."""

    def test_initialization(self, sample_config):
        """Test trainer initialization."""
        trainer = CurriculumTrainer(sample_config)

        assert trainer.config is not None
        assert trainer.scheduler is not None
        assert trainer.data_loader is not None
        assert trainer.metrics_logger is not None
        assert trainer.checkpoint_dir.exists()

    def test_checkpoint_directory_creation(self, sample_config, tmp_path):
        """Test checkpoint directory creation."""
        sample_config["system"]["checkpointing"]["save_dir"] = str(tmp_path / "checkpoints")

        trainer = CurriculumTrainer(sample_config)

        assert trainer.checkpoint_dir.exists()
        assert trainer.checkpoint_dir.is_dir()

    def test_seed_setting(self, sample_config):
        """Test random seed setting."""
        seed = sample_config["system"]["seed"]

        trainer1 = CurriculumTrainer(sample_config)
        random_value1 = torch.rand(1).item()

        trainer2 = CurriculumTrainer(sample_config)
        random_value2 = torch.rand(1).item()

        # With same seed, should get same random values
        assert random_value1 == random_value2

    def test_save_checkpoint(self, sample_config, tmp_path, device):
        """Test checkpoint saving."""
        from offline_robotic_manipulation_curriculum.models.model import CQLAgent

        sample_config["system"]["checkpointing"]["save_dir"] = str(tmp_path / "checkpoints")

        trainer = CurriculumTrainer(sample_config)

        # Initialize a dummy agent
        trainer.agent = CQLAgent(obs_dim=24, act_dim=4).to(device)
        trainer.optimizer = torch.optim.Adam(trainer.agent.parameters())

        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint.pt")

        checkpoint_path = trainer.checkpoint_dir / "test_checkpoint.pt"
        assert checkpoint_path.exists()

        # Load and verify
        checkpoint = torch.load(checkpoint_path, map_location=device)
        assert "agent_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "config" in checkpoint

    def test_device_selection(self, sample_config):
        """Test device selection."""
        trainer = CurriculumTrainer(sample_config)

        # Device should be either cuda or cpu
        assert trainer.device in [torch.device("cuda"), torch.device("cpu")]


def test_integration_curriculum_flow(sample_config):
    """Integration test for curriculum flow."""
    stages = [
        {"name": "stage1", "difficulty": 1, "env_name": "env1", "success_threshold": 0.4, "min_episodes": 100},
        {"name": "stage2", "difficulty": 2, "env_name": "env2", "success_threshold": 0.5, "min_episodes": 100},
        {"name": "stage3", "difficulty": 3, "env_name": "env3", "success_threshold": 0.6, "min_episodes": 100},
    ]

    scheduler = CurriculumScheduler(
        stages=stages,
        mode="performance_based",
        min_stage_epochs=2,
        max_stage_epochs=5,
        progression_patience=2,
    )

    # Start at first stage
    assert scheduler.current_stage_idx == 0

    # Simulate training
    for epoch in range(3):
        scheduler.stage_epochs += 1

    # Good performance - should progress
    should_progress = scheduler.should_progress(
        epoch=3,
        metrics={"normalized_return": 0.5},
    )
    assert should_progress

    # Progress to next stage
    scheduler.progress_stage()
    assert scheduler.current_stage_idx == 1

    # Simulate more training
    for epoch in range(3):
        scheduler.stage_epochs += 1

    # Progress again
    should_progress = scheduler.should_progress(
        epoch=3,
        metrics={"normalized_return": 0.6},
    )
    assert should_progress

    scheduler.progress_stage()
    assert scheduler.current_stage_idx == 2

    # At final stage - should not progress
    scheduler.stage_epochs = 10
    should_progress = scheduler.should_progress(
        epoch=10,
        metrics={"normalized_return": 0.7},
    )
    assert not should_progress
