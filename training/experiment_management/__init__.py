"""
Experiment Management Subsystem.

Provides tools for discovering, loading, ranking, and generating leaderboards
across multiple CrowdDNA experiments.
"""

from training.experiment_management.models import Experiment
from training.experiment_management.discovery import ExperimentDiscovery
from training.experiment_management.validator import ArtifactValidator
from training.experiment_management.metrics import MetricExtractor
from training.experiment_management.loader import ExperimentLoader
from training.experiment_management.ranking import RankingStrategy, DefaultRankingStrategy
from training.experiment_management.comparator import ExperimentComparator
from training.experiment_management.leaderboard import LeaderboardGenerator
from training.experiment_management.plotting import PlotGenerator, register_plot
from training.experiment_management.cli import cli_main

__all__ = [
    "Experiment",
    "ExperimentDiscovery",
    "ArtifactValidator",
    "MetricExtractor",
    "ExperimentLoader",
    "RankingStrategy",
    "DefaultRankingStrategy",
    "ExperimentComparator",
    "LeaderboardGenerator",
    "PlotGenerator",
    "register_plot",
    "cli_main",
]
