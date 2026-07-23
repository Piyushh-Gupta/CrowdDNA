"""
Test suite for the experiment management subsystem.
"""
import json
import pytest
from pathlib import Path

from training.experiment_management.models import Experiment
from training.experiment_management.discovery import ExperimentDiscovery
from training.experiment_management.validator import ArtifactValidator
from training.experiment_management.metrics import MetricExtractor
from training.experiment_management.loader import ExperimentLoader
from training.experiment_management.ranking import DefaultRankingStrategy
from training.experiment_management.comparator import ExperimentComparator
from training.experiment_management.leaderboard import LeaderboardGenerator
from training.experiment_management.plotting import PlotGenerator


@pytest.fixture
def mock_exp_dir(tmp_path: Path):
    """Creates a valid mock experiment directory."""
    exp_dir = tmp_path / "baseline_test"
    exp_dir.mkdir()
    
    (exp_dir / "metadata").mkdir()
    with open(exp_dir / "metadata" / "experiment_metadata.json", "w") as f:
        json.dump({"timestamp": "2026-07-23", "git_commit": "abc1234"}, f)
        
    (exp_dir / "checkpoints").mkdir()
    (exp_dir / "checkpoints" / "best.pt").touch()
    
    (exp_dir / "analysis").mkdir()
    (exp_dir / "analysis" / "raw").mkdir()
    with open(exp_dir / "analysis" / "raw" / "analysis.json", "w") as f:
        json.dump({
            "performance": {
                "f1_macro": 0.85,
                "accuracy": 0.90,
                "precision_macro": 0.80,
                "recall_macro": 0.82
            }
        }, f)
        
    (exp_dir / "deploy").mkdir()
    with open(exp_dir / "deploy" / "deployment_validation.md", "w") as f:
        f.write("**TorchScript via InferenceRuntime**: 5.2 ms\n")
        f.write("**TorchScript**: 1.5 MB\n")
        
    (exp_dir / "logs").mkdir()
    with open(exp_dir / "logs" / "training.log", "w") as f:
        f.write("Total training time: 3600s\nPeak GPU allocated: 2000 MB\n")

    return exp_dir


def test_validator_success(mock_exp_dir):
    assert ArtifactValidator.is_valid_experiment(mock_exp_dir)


def test_validator_missing_checkpoint(tmp_path):
    exp_dir = tmp_path / "bad_test"
    exp_dir.mkdir()
    (exp_dir / "metadata").mkdir()
    (exp_dir / "metadata" / "experiment_metadata.json").touch()
    
    # Missing checkpoints directory
    assert not ArtifactValidator.is_valid_experiment(exp_dir)


def test_metric_extractor(mock_exp_dir):
    config, metrics, deployment, hardware = MetricExtractor.extract_all(mock_exp_dir)
    assert metrics["f1"] == 0.85
    assert metrics["accuracy"] == 0.90
    assert deployment["latency_ms"] == 5.2
    assert deployment["size_mb"] == 1.5
    assert metrics["training_time"] == 3600.0
    assert hardware["peak_gpu_allocated_mb"] == 2000.0


def test_discovery_and_loader(mock_exp_dir):
    discovery = ExperimentDiscovery(base_dir=str(mock_exp_dir.parent))
    loader = ExperimentLoader(discovery)
    experiments = loader.load_all()
    
    assert len(experiments) == 1
    exp = experiments[0]
    assert exp.name == "baseline_test"
    assert exp.timestamp == "2026-07-23"
    assert exp.metrics["f1"] == 0.85


def test_ranking_ties_and_determinism():
    comparator = ExperimentComparator(DefaultRankingStrategy())
    
    exp1 = Experiment("exp1", "", "", "", metrics={"f1": 0.8, "accuracy": 0.9}, deployment={"latency_ms": 10})
    exp2 = Experiment("exp2", "", "", "", metrics={"f1": 0.8, "accuracy": 0.9}, deployment={"latency_ms": 5}) # Wins tie via latency
    exp3 = Experiment("exp3", "", "", "", metrics={"f1": 0.9, "accuracy": 0.8}, deployment={"latency_ms": 20}) # Wins overall via F1
    
    ranked = comparator.compare([exp1, exp2, exp3])
    
    assert ranked[0].name == "exp3"
    assert ranked[1].name == "exp2"
    assert ranked[2].name == "exp1"
    
    assert ranked[0].rank == 1
    assert ranked[2].rank == 3


def test_missing_metrics_fallback(tmp_path):
    """Test when analysis.json exists but is missing fields, or deploy report is missing."""
    exp_dir = tmp_path / "fallback_test"
    exp_dir.mkdir()
    (exp_dir / "metadata").mkdir()
    (exp_dir / "checkpoints").mkdir()
    (exp_dir / "checkpoints" / "best.pt").touch()
    with open(exp_dir / "metadata" / "experiment_metadata.json", "w") as f:
        json.dump({}, f)
        
    config, metrics, deployment, hardware = MetricExtractor.extract_all(exp_dir)
    assert metrics["f1"] == "N/A"
    assert deployment["latency_ms"] == "N/A"


def test_leaderboard_generation(tmp_path):
    exp1 = Experiment("exp1", "/path1", "2026", "abc", metrics={"f1": 0.9}, deployment={"latency_ms": 10}, hardware={}, rank=1)
    
    LeaderboardGenerator.generate_json([exp1], tmp_path)
    LeaderboardGenerator.generate_csv([exp1], tmp_path)
    LeaderboardGenerator.generate_markdown([exp1], tmp_path)
    
    assert (tmp_path / "leaderboard.json").exists()
    assert (tmp_path / "leaderboard.csv").exists()
    assert (tmp_path / "leaderboard.md").exists()
    
    with open(tmp_path / "leaderboard.json", "r") as f:
        data = json.load(f)
        assert data["_meta"]["schema_version"] == "1.0.0"
        assert len(data["leaderboard"]) == 1
        assert data["leaderboard"][0]["F1 Score"] == 0.9


def test_plot_generator_empty(tmp_path):
    """Ensure it doesn't crash on empty experiments."""
    PlotGenerator.generate_plots([], tmp_path)
    assert (tmp_path / "plots").exists()
