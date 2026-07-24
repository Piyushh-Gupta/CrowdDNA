"""
Defines strategies for ranking experiments.
"""
from abc import ABC, abstractmethod
from typing import List

from training.experiment_management.models import Experiment


class RankingStrategy(ABC):
    """Abstract base class for ranking strategies."""
    
    @abstractmethod
    def sort(self, experiments: List[Experiment]) -> List[Experiment]:
        """Sorts the experiments based on the strategy."""
        pass


class DefaultRankingStrategy(RankingStrategy):
    """
    Default ranking:
    1. Primary: F1 Score (Desc)
    2. Secondary: Accuracy (Desc)
    3. Tertiary: Deployment Latency (Asc)
    """

    def sort(self, experiments: List[Experiment]) -> List[Experiment]:
        def sort_key(exp: Experiment):
            # Extract F1, safely fall back to 0.0 if N/A
            f1 = exp.metrics.get("f1", "N/A")
            f1_val = float(f1) if f1 != "N/A" else -1.0
            
            # Extract Accuracy
            acc = exp.metrics.get("accuracy", "N/A")
            acc_val = float(acc) if acc != "N/A" else -1.0
            
            # Extract Latency (Ascending means smaller is better, so negate for descending sort)
            lat = exp.deployment.get("latency_ms", "N/A")
            lat_val = float(lat) if lat != "N/A" else float("inf")
            
            # We sort descending overall, so we negate latency
            # Finally, we use exp.name (ascending) as a tie-breaker.
            # Since overall is reverse=True, we negate the string comparison? No, python sorting handles tuples.
            # Wait, reverse=True reverses the entire tuple comparison. 
            # So to make exp.name ascending when reverse=True, we need to negate it?
            # Strings cannot be negated. We can sort without reverse=True and negate the primary metrics instead.
            return (-f1_val, -acc_val, lat_val, exp.name)

        return sorted(experiments, key=sort_key)


# Registry for easy CLI access
STRATEGIES = {
    "default": DefaultRankingStrategy()
}
