"""Logging configuration and utilities."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    format: Optional[str] = None
) -> None:
    """
    Set up logging configuration.

    Parameters
    ----------
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : str, optional
        If provided, also log to this file
    format : str, optional
        Custom format string
    """
    # Default format
    if format is None:
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format))
    root_logger.addHandler(console_handler)

    # File handler if requested
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format))
        root_logger.addHandler(file_handler)

def get_experiment_logger(name: str) -> logging.Logger:
    """Get a logger for a specific experiment."""
    return logging.getLogger(f'geomimb.experiments.{name}')

class ProgressLogger:
    """Context manager for logging progress of long-running operations."""

    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        self.operation = operation
        self.logger = logger or logging.getLogger()
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} (took {duration:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation} (after {duration:.2f}s)")

def log_dict(d: dict, logger: Optional[logging.Logger] = None, prefix: str = "") -> None:
    """Log dictionary contents in a readable format."""
    logger = logger or logging.getLogger()
    for key, value in d.items():
        if isinstance(value, float):
            logger.info(f"{prefix}{key}: {value:.6f}")
        else:
            logger.info(f"{prefix}{key}: {value}")

def log_separator(logger: Optional[logging.Logger] = None, char: str = '-', length: int = 60) -> None:
    """Log a separator line."""
    logger = logger or logging.getLogger()
    logger.info(char * length)

def log_experiment_start(experiment_name: str, logger: Optional[logging.Logger] = None) -> None:
    """Log the start of an experiment with nice formatting."""
    logger = logger or logging.getLogger()
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Starting: {experiment_name}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info("")