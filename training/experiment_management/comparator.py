"""
Comparator module for ranking and comparing experiments.
"""
from dataclasses import replace
from typing import List, Optional

from training.experiment_management.models import Experiment
from training.experiment_management.ranking import RankingStrategy


class ExperimentComparator:
    """Aggregates and ranks experiments."""

    def __init__(self, strategy: RankingStrategy):
        self.strategy = strategy

    def compare(self, experiments: List[Experiment], top_n: Optional[int] = None) -> List[Experiment]:
        """
        Ranks the experiments and optionally limits the output to top_n.
        """
        # Sort based on strategy
        sorted_exps = self.strategy.sort(experiments)
        
        # Limit to top_n if requested
        if top_n is not None and top_n > 0:
            sorted_exps = sorted_exps[:top_n]
            
        # Assign ranks via replace (dataclass is immutable)
        ranked_exps = []
        for i, exp in enumerate(sorted_exps):
            ranked_exps.append(replace(exp, rank=i+1))
            
        return ranked_exps
