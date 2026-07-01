# =============================================================================
# src/utils/logger.py
# Centralised logging — every script imports this instead of using print()
# =============================================================================

import logging
import sys
from pathlib import Path

def get_logger(name: str, log_file: str = "logs/pipeline.log") -> logging.Logger:
    """
    Returns a configured logger that writes to both console and a log file.
    
    Args:
        name     : Usually __name__ from the calling module
        log_file : Path to the log file (created automatically)
    
    Returns:
        logging.Logger instance
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if function is called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # File handler — DEBUG and above (full detail)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
