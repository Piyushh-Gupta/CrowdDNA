"""
Plotting Registry and Base Protocol.

Provides a registry for plotting generators.
"""
from abc import ABC, abstractmethod
from typing import Dict, Type, List

from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.models import ExplanationGraph

class PlotterProtocol(ABC):
    """Protocol for all plotters."""
    
    @abstractmethod
    def plot(self, session: ExplanationSession, graphs: List[ExplanationGraph]) -> None:
        """Generates plots from the given ExplanationGraphs."""
        pass

class PlottingRegistry:
    """Registry for managing and instantiating plotters."""
    
    _registry: Dict[str, Type[PlotterProtocol]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(plotter_cls: Type[PlotterProtocol]):
            if name in cls._registry:
                raise ValueError(f"Plotter '{name}' is already registered.")
            cls._registry[name] = plotter_cls
            return plotter_cls
        return wrapper

    @classmethod
    def get_plotters(cls) -> List[Type[PlotterProtocol]]:
        return list(cls._registry.values())
