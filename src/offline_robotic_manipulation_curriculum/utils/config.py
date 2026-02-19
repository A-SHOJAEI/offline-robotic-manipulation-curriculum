"""Configuration management utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save.
        output_path: Path where to save the configuration.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def get_device(config: Dict[str, Any]) -> str:
    """Determine device to use based on configuration and availability.

    Args:
        config: Configuration dictionary.

    Returns:
        Device string ('cuda' or 'cpu').
    """
    import torch

    device_config = config.get("system", {}).get("device", "auto")
    use_gpu = config.get("system", {}).get("use_gpu", True)

    if device_config == "auto":
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU")
    else:
        device = device_config
        logger.info(f"Using device: {device}")

    return device


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration has required fields.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If required fields are missing.
    """
    required_sections = ["curriculum", "model", "training", "evaluation", "system"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate curriculum stages
    if "stages" not in config["curriculum"]:
        raise ValueError("Curriculum must define stages")

    if len(config["curriculum"]["stages"]) == 0:
        raise ValueError("Curriculum must have at least one stage")

    # Validate model parameters
    model_config = config["model"]
    if "architecture" not in model_config:
        raise ValueError("Model architecture must be specified")

    if model_config["architecture"] == "cql" and "cql" not in model_config:
        raise ValueError("CQL architecture requires cql configuration parameters")

    logger.info("Configuration validation passed")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override configuration into base configuration.

    Args:
        base_config: Base configuration dictionary.
        override_config: Configuration values to override.

    Returns:
        Merged configuration dictionary.
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
