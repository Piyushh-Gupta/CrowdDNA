"""
Orchestrates experiment discovery, validation, and metric extraction.
"""
import json
import logging
from typing import List

from training.experiment_management.discovery import ExperimentDiscovery
from training.experiment_management.metrics import MetricExtractor
from training.experiment_management.models import Experiment

logger = logging.getLogger(__name__)


class ExperimentLoader:
    """Loads Experiment objects from the filesystem."""

    def __init__(self, discovery: ExperimentDiscovery):
        self.discovery = discovery

    def load_all(self, include_incomplete: bool = False) -> List[Experiment]:
        """
        Discovers, validates, and extracts metrics for all experiments.
        """
        paths = self.discovery.discover(include_incomplete=include_incomplete)
        experiments = []

        for path in paths:
            try:
                exp = self._load_single(path)
                experiments.append(exp)
            except Exception as e:
                logger.error(f"Failed to load experiment {path.name}: {e}")

        return experiments

    def _load_single(self, exp_dir) -> Experiment:
        config, metrics, deployment, hardware = MetricExtractor.extract_all(exp_dir)
        
        # Grab timestamp and commit from metadata if possible
        timestamp = "N/A"
        git_commit = "N/A"
        meta_path = exp_dir / "metadata" / "experiment_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    timestamp = data.get("timestamp", "N/A")
                    git_commit = data.get("git_commit", "N/A")
            except Exception:
                pass

        return Experiment(
            name=exp_dir.name,
            path=str(exp_dir),
            timestamp=timestamp,
            git_commit=git_commit,
            configuration=config,
            metrics=metrics,
            deployment=deployment,
            hardware=hardware
        )
