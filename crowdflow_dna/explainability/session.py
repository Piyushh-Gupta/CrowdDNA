"""
Explanation Session data model.

This module defines the immutable ExplanationSession which represents
a complete execution session independently of individual explanations.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ExplanationSession:
    """Immutable data model for an explainability execution session."""
    session_id: str
    crowddna_version: str
    schema_version: str
    deployment_model_hash: str
    deployment_artifact_hash: str
    dataset_identifier: str
    enabled_explainers: tuple[str, ...]
    configuration_hash: str
    execution_timestamp: str
    execution_duration: float
    output_directory: str
    cache_status: str
    random_seed: Optional[int] = None
