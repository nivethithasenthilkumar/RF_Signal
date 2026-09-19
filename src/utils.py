"""
Utility functions for logging, file persistence, and formatting.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict


def setup_logger(name: str = "rf_signal_ml", level: int = logging.INFO) -> logging.Logger:
    """Configures a standardized console logger with structured formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_directories(*paths: Path) -> None:
    """Ensures directories exist, creating them if necessary."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Saves dictionary to JSON file with indentation."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Loads dictionary from JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
