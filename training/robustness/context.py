"""
CrowdFlow DNA — Evaluation Context
==================================
Module: training/robustness/context.py

Defines the immutable EvaluationContext object that encapsulates runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch

@dataclass(frozen=True)
class EvaluationContext:
    """
    Immutable context object containing all runtime state and configuration
    needed for robustness, stress, and regression evaluations.
    
    Attributes:
        deployment_model_path: Path to the TorchScript deployment artifact.
        dataset_provider: The DatasetProvider instance yielding sequences.
        random_seed: Master random seed for deterministic perturbations.
        protocol_name: Name of the active evaluation protocol.
        device: Torch device (cpu or cuda).
        output_directory: Directory where reports and plots will be saved.
        config: Full system configuration dictionary.
        experiment_metadata: Dictionary containing metadata about the run.
    """
    deployment_model_path: Path
    dataset_provider: Any  # Avoid circular import with datasets.py
    random_seed: int
    protocol_name: str
    device: torch.device
    output_directory: Path
    config: Dict[str, Any] = field(default_factory=dict)
    experiment_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_output_path(self, filename: str) -> Path:
        """Helper to safely construct an output path within the protocol directory."""
        return self.output_directory / filename

