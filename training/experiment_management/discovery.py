"""
Discovery module for locating experiment runs.
"""
import logging
from pathlib import Path
from typing import List

from training.experiment_management.validator import ArtifactValidator

logger = logging.getLogger(__name__)

class ExperimentDiscovery:
    """Discovers experiment directories."""

    def __init__(self, base_dir: str = "experiments/runs"):
        self.base_dir = Path(base_dir)

    def discover(self, include_incomplete: bool = False) -> List[Path]:
        """
        Scans the base directory for experiments.
        
        Args:
            include_incomplete: If True, bypasses ArtifactValidator.
        Returns:
            List of valid experiment Paths.
        """
        if not self.base_dir.exists():
            logger.warning(f"Base directory {self.base_dir} does not exist.")
            return []

        discovered = []
        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            if include_incomplete or ArtifactValidator.is_valid_experiment(exp_dir):
                discovered.append(exp_dir)

        return discovered
