"""
Tests for the generalized experiment analysis framework.
"""

import json
from pathlib import Path

import pytest

from training.analyze_experiment import analyze_experiment


@pytest.fixture
def mock_experiment_dir(tmp_path: Path) -> Path:
    """Sets up a mock experiment directory with required artifacts."""
    exp_dir = tmp_path / "experiments" / "runs" / "test_exp_001"
    
    # Create subdirs
    (exp_dir / "metadata").mkdir(parents=True)
    (exp_dir / "metrics").mkdir(parents=True)
    (exp_dir / "checkpoints").mkdir(parents=True)
    (exp_dir / "reports").mkdir(parents=True)
    
    # Mock metadata
    meta = {
        "configuration_snapshot": {
            "model": {
                "gnn_hidden_dim": 128,
                "gru_hidden_dim": 64,
                "gru_num_layers": 2,
                "gru_bidirectional": True,
                "classes": ["Safe", "Critical"]
            },
            "training": {
                "sequence_length": 5
            }
        },
        "total_parameters": 1000,
        "trainable_parameters": 1000,
        "training_duration": 100.0,
        "evaluation_duration": 10.0,
        "dataset_hash": "abcd123",
        "manifest_version": "v1"
    }
    with open(exp_dir / "metadata" / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
        
    # Mock metrics
    history = {
        "train_loss": [1.0, 0.5],
        "val_loss": [1.2, 0.6],
        "train_acc": [0.5, 0.8],
        "val_acc": [0.4, 0.7]
    }
    with open(exp_dir / "metrics" / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f)
        
    eval_metrics = {
        "accuracy": 0.9,
        "precision_macro": 0.85,
        "recall_macro": 0.88,
        "f1_macro": 0.86,
        "targets": [0, 1, 0, 1],
        "predictions": [0, 1, 0, 0],
        "probabilities": [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.6, 0.4]
        ]
    }
    with open(exp_dir / "metrics" / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f)
        
    # Mock deployment validation (parent.parent)
    dep_dir = tmp_path / "experiments" / "deployment_validation"
    dep_dir.mkdir(parents=True)
    with open(dep_dir / "deployment_validation.md", "w", encoding="utf-8") as f:
        f.write("**TorchScript via InferenceRuntime**: 5.89 ms\n")
        f.write("**TorchScript**: 1.71 MB\n")
        f.write("**ONNX**: 0.0 MB\n")
        
    # Mock checkpoint file
    ckpt = exp_dir / "checkpoints" / "best.pt"
    ckpt.write_text("dummy binary data")
    
    # Mock existing reports plot
    (exp_dir / "reports" / "confusion_matrix.png").write_text("fake png")
    
    return exp_dir


def test_analyze_experiment_deterministic_outputs(mock_experiment_dir: Path) -> None:
    """Verifies that analyze_experiment produces the correct directory structure and files."""
    analyze_experiment(mock_experiment_dir)
    
    analysis_dir = mock_experiment_dir / "analysis"
    assert analysis_dir.exists()
    
    # Check dirs
    assert (analysis_dir / "report").exists()
    assert (analysis_dir / "plots").exists()
    assert (analysis_dir / "tables").exists()
    assert (analysis_dir / "raw").exists()
    
    # Check files
    assert (analysis_dir / "raw" / "analysis.json").exists()
    assert (analysis_dir / "tables" / "summary.csv").exists()
    assert (analysis_dir / "report" / "benchmark_analysis.md").exists()
    
    # Check plots (generated)
    assert (analysis_dir / "plots" / "loss_curve.png").exists()
    assert (analysis_dir / "plots" / "accuracy_curve.png").exists()
    assert (analysis_dir / "plots" / "confidence_histogram.png").exists()
    assert (analysis_dir / "plots" / "class_distribution.png").exists()
    
    # Check reused plot
    assert (analysis_dir / "plots" / "confusion_matrix.png").exists()
    assert (analysis_dir / "plots" / "confusion_matrix.png").read_text() == "fake png"
    
    # Validate JSON contents
    with open(analysis_dir / "raw" / "analysis.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["experiment_name"] == "test_exp_001"
    assert data["model_metadata"]["parameter_count"] == 1000
    assert data["model_metadata"]["gat_hidden_dim"] == 128
    assert data["dataset_summary"]["validation_samples"] == 4
    assert data["dataset_summary"]["class_distribution"]["Safe"] == 2
    assert "mean" in data["statistics"]["confidence"]
    assert "median" in data["statistics"]["confidence"]
    assert data["performance"]["latency"] == 5.89


def test_analyze_experiment_missing_artifacts(tmp_path: Path) -> None:
    """Verifies analyze_experiment handles missing optional artifacts gracefully."""
    exp_dir = tmp_path / "experiments" / "runs" / "empty_exp"
    exp_dir.mkdir(parents=True)
    
    # Run analysis on basically empty dir
    analyze_experiment(exp_dir)
    
    analysis_dir = exp_dir / "analysis"
    assert (analysis_dir / "raw" / "analysis.json").exists()
    
    with open(analysis_dir / "raw" / "analysis.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["performance"]["latency"] == "N/A"
    assert data["model_metadata"]["gat_hidden_dim"] == "N/A"
    assert data["dataset_summary"]["validation_samples"] == 0
    
    # Summary CSV and Markdown should still exist
    assert (analysis_dir / "tables" / "summary.csv").exists()
    assert (analysis_dir / "report" / "benchmark_analysis.md").exists()
