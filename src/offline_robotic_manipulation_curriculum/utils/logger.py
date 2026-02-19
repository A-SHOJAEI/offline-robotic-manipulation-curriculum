"""Logging utilities for training and evaluation."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Setup logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_file: Optional path to log file.
        level: Logging level.
        format_string: Optional custom format string.

    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """Logger for tracking training and evaluation metrics."""

    def __init__(self, log_dir: str, use_mlflow: bool = True, use_tensorboard: bool = True):
        """Initialize metrics logger.

        Args:
            log_dir: Directory for storing logs.
            use_mlflow: Whether to use MLflow tracking.
            use_tensorboard: Whether to use TensorBoard.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_mlflow = use_mlflow
        self.use_tensorboard = use_tensorboard

        self.logger = logging.getLogger(__name__)

        # Initialize TensorBoard writer if enabled
        self.tb_writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                self.logger.info("TensorBoard logging enabled")
            except ImportError:
                self.logger.warning("TensorBoard not available, skipping")
                self.use_tensorboard = False

    def log_metrics(self, metrics: dict, step: int, prefix: str = "") -> None:
        """Log metrics to configured backends.

        Args:
            metrics: Dictionary of metric names and values.
            step: Current training step.
            prefix: Optional prefix for metric names.
        """
        # Log to console
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {prefix}{metric_str}")

        # Log to MLflow
        if self.use_mlflow:
            try:
                import mlflow
                for name, value in metrics.items():
                    mlflow.log_metric(f"{prefix}{name}", value, step=step)
            except Exception as e:
                self.logger.warning(f"MLflow logging failed: {e}")

        # Log to TensorBoard
        if self.use_tensorboard and self.tb_writer is not None:
            try:
                for name, value in metrics.items():
                    self.tb_writer.add_scalar(f"{prefix}{name}", value, step)
            except Exception as e:
                self.logger.warning(f"TensorBoard logging failed: {e}")

    def log_hyperparameters(self, params: dict) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of hyperparameter names and values.
        """
        if self.use_mlflow:
            try:
                import mlflow
                mlflow.log_params(params)
            except Exception as e:
                self.logger.warning(f"MLflow parameter logging failed: {e}")

    def close(self) -> None:
        """Close all logging backends."""
        if self.tb_writer is not None:
            self.tb_writer.close()
