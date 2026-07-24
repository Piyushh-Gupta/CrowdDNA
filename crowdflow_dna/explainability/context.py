"""
Explanation Context data model.

This module defines the immutable ExplanationContext which is passed
throughout the explainability pipeline.
"""
from dataclasses import dataclass
from typing import Any, Dict
import logging

from crowdflow_dna.inference.runtime import InferenceRuntime
from crowdflow_dna.model.deployment_model import CrowdDNADeploymentModel
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ExplanationContext:
    """Immutable data model for an explainability execution context."""
    model: CrowdDNADeploymentModel
    runtime: InferenceRuntime
    dataset_provider: Dataset
    output_directory: str
    configuration: Dict[str, Any]
    logger: logging.Logger
    device: str
