import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from training.evaluate_model import EvaluationResult
from training.run_experiment import ArtifactManager, ExperimentResult, ExperimentRunner
from training.train_model import TrainingHistory


@pytest.fixture
def mock_dataset_and_config(tmp_path):
    """Creates a temporary dataset and model configuration."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    classes = ["Safe", "Congesting", "Critical"]
    manifest = []
    
    # 4 trajectories (min required for batch=4 or splitting)
    for i in range(4):
        traj = {
            "sequence_id": f"seq_{i}",
            "risk_class": classes[i % 3],
            "positions": [[[0.0, 0.0]]],
            "velocities": [[[0.0, 0.0]]],
            "frame_labels": [classes[i % 3]]
        }
        fname = f"traj_{i}.json"
        with open(data_dir / fname, "w") as f:
            json.dump(traj, f)
        manifest.append({"file_path": fname, "num_timesteps": 1})
        
    with open(data_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    config = {
        "model": {
            "gnn_hidden_dim": 8,
            "gru_hidden_dim": 8,
            "gru_num_layers": 1,
            "gru_dropout": 0.0,
            "gru_bidirectional": False,
            "num_gnn_layers": 1,
            "classes": classes,
        },
        "simulation": {
            "output_dir": str(data_dir)
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 2,
            "validation_split": 0.25,
            "random_seed": 42,
            "deterministic": True,
        }
    }
    
    config_path = tmp_path / "mock_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    return config_path, str(tmp_path)


def test_artifact_manager_layout(tmp_path):
    manager = ArtifactManager(str(tmp_path / "experiment"))
    
    assert manager.config_dir.exists()
    assert manager.checkpoints_dir.exists()
    assert manager.metrics_dir.exists()
    assert manager.metadata_dir.exists()
    assert manager.logs_dir.exists()


def test_artifact_manager_saving(tmp_path):
    manager = ArtifactManager(str(tmp_path / "experiment"))
    
    # Save config
    manager.save_config({"test": 123})
    assert (manager.config_dir / "snapshot.yaml").exists()
    
    # Save history
    hist = TrainingHistory([0.1], [0.2], [0.8], [0.9], 0.2, 1)
    manager.save_training_history(hist)
    assert (manager.metrics_dir / "training_history.json").exists()
    
    # Save eval
    eval_res = EvaluationResult(
        accuracy=1.0,
        precision_macro=1.0,
        recall_macro=1.0,
        f1_macro=1.0,
        precision_per_class=np.array([1.0]),
        recall_per_class=np.array([1.0]),
        f1_per_class=np.array([1.0]),
        confusion_matrix=np.array([[1]]),
        normalized_confusion_matrix=np.array([[1.0]]),
        predictions=np.array([0]),
        targets=np.array([0]),
        probabilities=np.array([[1.0]]),
    )
    manager.save_evaluation_metrics(eval_res)
    assert (manager.metrics_dir / "evaluation_metrics.json").exists()
    
    # Check NumpyEncoder serialization
    with open(manager.metrics_dir / "evaluation_metrics.json") as f:
        data = json.load(f)
        assert data["accuracy"] == 1.0
        assert data["confusion_matrix"] == [[1]]


def test_experiment_runner_execution(mock_dataset_and_config):
    config_path, tmp_path = mock_dataset_and_config
    exp_dir = Path(tmp_path) / "exp1"
    
    runner = ExperimentRunner(str(config_path), str(exp_dir))
    result = runner.run()
    
    assert isinstance(result, ExperimentResult)
    assert result.total_training_time > 0
    assert result.evaluation_time > 0
    assert result.experiment_directory == exp_dir
    assert (exp_dir / "checkpoints" / "best.pt").exists()
    assert (exp_dir / "metrics" / "training_history.json").exists()
    
    # Check deterministic execution
    exp_dir2 = Path(tmp_path) / "exp2"
    runner2 = ExperimentRunner(str(config_path), str(exp_dir2))
    result2 = runner2.run()
    
    # Timings will differ, but metrics should be identical
    assert result.evaluation_result.accuracy == result2.evaluation_result.accuracy
    assert np.array_equal(result.evaluation_result.predictions, result2.evaluation_result.predictions)
