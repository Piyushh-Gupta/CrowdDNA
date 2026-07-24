"""
CrowdFlow DNA — Robustness Logger
=================================
Module: training/robustness/logger.py

Dedicated robustness logger that creates deterministic, structured logs suitable for debugging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

def setup_robustness_logger(log_file_path: Path | None = None) -> logging.Logger:
    """Sets up and returns a dedicated logger for the robustness framework."""
    logger = logging.getLogger("crowddna.robustness")
    logger.setLevel(logging.DEBUG)
    
    # Prevent propagation to root logger to avoid duplicate prints
    logger.propagate = False
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler if specified
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
