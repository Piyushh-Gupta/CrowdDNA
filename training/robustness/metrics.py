"""
CrowdFlow DNA — Metrics & Scoring
=================================
Module: training/robustness/metrics.py

Pluggable metric interface, metric registry, and robustness scoring strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Type

import numpy as np

class Metric(ABC):
    """Abstract interface for robustness evaluation metrics."""
    
    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state of the metric."""
        pass

    @abstractmethod
    def update(
        self, 
        predictions: np.ndarray, 
        probabilities: np.ndarray, 
        targets: np.ndarray,
        inference_time_ms: float
    ) -> None:
        """Accumulates metrics from a single batch."""
        pass
        
    @abstractmethod
    def compute(self) -> float:
        """Computes and returns the final metric value."""
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """Returns the canonical name of the metric."""
        pass


METRIC_REGISTRY: Dict[str, Type[Metric]] = {}

def register_metric(name: str) -> Callable:
    def wrapper(cls: Type[Metric]) -> Type[Metric]:
        METRIC_REGISTRY[name] = cls
        return cls
    return wrapper


@register_metric("accuracy")
class AccuracyMetric(Metric):
    def __init__(self) -> None:
        self.reset()
        
    def reset(self) -> None:
        self.correct = 0
        self.total = 0
        
    def update(
        self, predictions: np.ndarray, probabilities: np.ndarray, targets: np.ndarray, inference_time_ms: float
    ) -> None:
        self.correct += int((predictions == targets).sum())
        self.total += len(targets)
        
    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0
        
    def get_name(self) -> str:
        return "Accuracy"


@register_metric("latency")
class LatencyMetric(Metric):
    def __init__(self) -> None:
        self.reset()
        
    def reset(self) -> None:
        self.total_time_ms = 0.0
        self.batches = 0
        
    def update(
        self, predictions: np.ndarray, probabilities: np.ndarray, targets: np.ndarray, inference_time_ms: float
    ) -> None:
        self.total_time_ms += inference_time_ms
        self.batches += 1
        
    def compute(self) -> float:
        return self.total_time_ms / self.batches if self.batches > 0 else 0.0
        
    def get_name(self) -> str:
        return "Average Latency (ms)"


@register_metric("confidence")
class ConfidenceShiftMetric(Metric):
    def __init__(self) -> None:
        self.reset()
        
    def reset(self) -> None:
        self.confidences: List[float] = []
        
    def update(
        self, predictions: np.ndarray, probabilities: np.ndarray, targets: np.ndarray, inference_time_ms: float
    ) -> None:
        # Confidence is the max probability for the predicted class
        confs = np.max(probabilities, axis=-1)
        self.confidences.extend(confs.tolist())
        
    def compute(self) -> float:
        return float(np.mean(self.confidences)) if self.confidences else 0.0
        
    def get_name(self) -> str:
        return "Mean Confidence"


class RobustnessScoringStrategy(ABC):
    """Computes a single aggregated robustness score from raw metrics."""
    
    @abstractmethod
    def compute(self, baseline_metrics: Dict[str, float], perturbed_metrics: Dict[str, float]) -> float:
        pass


class DefaultScoringStrategy(RobustnessScoringStrategy):
    """
    Computes robustness as the ratio of perturbed accuracy to baseline accuracy,
    penalized by latency increases.
    """
    def compute(self, baseline_metrics: Dict[str, float], perturbed_metrics: Dict[str, float]) -> float:
        base_acc = baseline_metrics.get("Accuracy", 1e-6)
        pert_acc = perturbed_metrics.get("Accuracy", 0.0)
        base_lat = baseline_metrics.get("Average Latency (ms)", 1.0)
        pert_lat = perturbed_metrics.get("Average Latency (ms)", 1.0)
        
        # Accuracy retention
        acc_retention = pert_acc / (base_acc + 1e-6)
        
        # Latency penalty (cap at 1.0)
        lat_penalty = min(1.0, base_lat / (pert_lat + 1e-6))
        
        return acc_retention * 0.8 + lat_penalty * 0.2
