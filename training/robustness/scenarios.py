"""
CrowdFlow DNA — Scenarios
=========================
Module: training/robustness/scenarios.py

High-level scenario definitions and registry for orchestrating perturbations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type

import torch
from torch_geometric.data import Data

from training.robustness import perturbations

class Scenario(ABC):
    """Abstract base class for testing scenarios."""
    
    def __init__(self, name: str, params: Dict[str, Any]) -> None:
        self.name = name
        self.params = params
        
    @abstractmethod
    def apply(self, sequence: List[Data], generator: torch.Generator) -> List[Data]:
        """Applies the scenario's perturbations to the sequence."""
        pass

# Global registry
SCENARIO_REGISTRY: Dict[str, Type[Scenario]] = {}

def register_scenario(name: str) -> Callable:
    """Decorator to register a scenario class in the registry."""
    def wrapper(cls: Type[Scenario]) -> Type[Scenario]:
        SCENARIO_REGISTRY[name] = cls
        return cls
    return wrapper


@register_scenario("missing_detections")
class MissingDetectionsScenario(Scenario):
    """Simulates missed detections by randomly dropping nodes."""
    def apply(self, sequence: List[Data], generator: torch.Generator) -> List[Data]:
        p = self.params.get("drop_prob", 0.1)
        return perturbations.drop_nodes(sequence, p, generator)


@register_scenario("gaussian_noise")
class GaussianNoiseScenario(Scenario):
    """Simulates sensor noise by adding Gaussian noise to node features."""
    def apply(self, sequence: List[Data], generator: torch.Generator) -> List[Data]:
        std = self.params.get("std", 0.1)
        return perturbations.add_gaussian_noise(sequence, std, generator)


@register_scenario("random_frame_drops")
class RandomFrameDropsScenario(Scenario):
    """Simulates dropped frames by replacing random frames with empty graphs."""
    def apply(self, sequence: List[Data], generator: torch.Generator) -> List[Data]:
        p = self.params.get("drop_prob", 0.1)
        return perturbations.drop_frames(sequence, p, generator)

