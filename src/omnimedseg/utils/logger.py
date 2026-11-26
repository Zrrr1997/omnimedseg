"""Logging configuration for the benchmark tool."""
import logging
import sys
from pathlib import Path
from datetime import datetime
import os


class PlainFormatter(logging.Formatter):
    """INFO: just the message. WARNING and above: 'warning: message' / 'error: message'."""

    def format(self, record):
        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.lower()}: {msg}"
        return msg


class VerboseFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")


def setup_logging(level=logging.INFO, log_file=None, verbose=False):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (optional)
        verbose: If True, set level to DEBUG
    """
    if verbose:
        level = logging.DEBUG
    
    # Plain messages by default; timestamps and logger names only in verbose mode.
    formatter = VerboseFormatter() if level <= logging.DEBUG else PlainFormatter()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler (stderr, so command output on stdout stays machine-readable)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name):
    """Get a logger instance."""
    return logging.getLogger(name)


# Create a default logger for the module
logger = get_logger(__name__)