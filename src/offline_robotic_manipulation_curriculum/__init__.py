"""Offline Robotic Manipulation Curriculum Learning Framework.

This package implements a curriculum learning approach for offline reinforcement
learning in robotic manipulation tasks. It combines Conservative Q-Learning with
automated curriculum scheduling to progressively transfer policies from simpler
to more complex tasks.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from offline_robotic_manipulation_curriculum.models.model import (
    CQLAgent,
    BehavioralCloningAgent,
    EnsembleQNetwork,
)
from offline_robotic_manipulation_curriculum.training.trainer import CurriculumTrainer
from offline_robotic_manipulation_curriculum.utils.config import load_config

__all__ = [
    "CQLAgent",
    "BehavioralCloningAgent",
    "EnsembleQNetwork",
    "CurriculumTrainer",
    "load_config",
]
