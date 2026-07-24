"""
Base Explainer Protocol.

Defines the ExplainerProtocol that all explanation algorithms must implement.
"""
from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Data

from crowdflow_dna.explainability.context import ExplanationContext
from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.models import ExplanationGraph


class ExplainerProtocol(ABC):
    """Protocol for all explanation algorithms."""
    
    @abstractmethod
    def explain(
        self,
        context: ExplanationContext,
        session: ExplanationSession,
        graph_sequence: Data,
        prediction: torch.Tensor,
        sequence_id: str
    ) -> ExplanationGraph:
        """Analyzes a model prediction and produces an ExplanationGraph."""
        pass
