"""
Explanation Graph and Versioned Schemas.

This module defines the canonical exchange object ExplanationGraph,
which is produced by explainers and consumed by reporters/plotters.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Any
import torch

from crowdflow_dna.explainability.session import ExplanationSession

@dataclass(frozen=True)
class ExplanationGraph:
    """The canonical exchange object for the explainability subsystem."""
    session: ExplanationSession
    explainer_name: str
    sequence_id: str
    node_importance: Optional[torch.Tensor]
    edge_importance: Optional[torch.Tensor]
    temporal_importance: Optional[torch.Tensor]
    confidence_analysis: Optional[Dict[str, float]]
    prediction_metadata: Dict[str, Any]
    saliency_tensors: Optional[Dict[str, torch.Tensor]] = None
    attention_tensors: Optional[Dict[str, torch.Tensor]] = None
