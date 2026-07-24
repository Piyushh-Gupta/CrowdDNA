"""
Base Reporter Protocol.

Defines the ReporterProtocol that all reporters must implement.
"""
from abc import ABC, abstractmethod
from typing import List

from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.models import ExplanationGraph


class ReporterProtocol(ABC):
    """Protocol for all reporters."""
    
    @abstractmethod
    def generate(self, session: ExplanationSession, graphs: List[ExplanationGraph]) -> None:
        """Generates a report from the given ExplanationGraphs."""
        pass
