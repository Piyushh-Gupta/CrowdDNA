import json
from pathlib import Path

import numpy as np
import pytest
import torch

from crowdflow_dna.graph.sequence_dataset import SequenceGraphDataset
from training.evaluate_model import EvaluationEngine, EvaluationResult, MetricsComputer
from training.train_model import CheckpointManager


@pytest.fixture
def mock_dataset_and_config(tmp_path):
    """Creates a temporary dataset and model configuration."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create 3 mini trajectories representing classes 0, 1, 2
    manifest = []
    classes = ["Safe", "Congesting", "Critical"]
    
    for i in range(3):
        traj = {
            "sequence_id": f"seq_{i}",
            "risk_class": classes[i],
            "positions": [[[0.0, 0.0]]],
            "velocities": [[[0.0, 0.0]]],
            "frame_labels": [classes[i]]
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
        }
    }
    
    dataset = SequenceGraphDataset(str(data_dir))
    return dataset, config, str(tmp_path)


def test_metrics_computer_correctness():
    """Test standard metrics are computed accurately on synthetic arrays."""
    computer = MetricsComputer(num_classes=3)
    
    targets = np.array([0, 1, 2, 0, 1, 2])
    predictions = np.array([0, 1, 1, 0, 1, 2])  # Missed one class 2 (predicted 1)
    probabilities = np.random.rand(6, 3) # dummy
    
    res = computer.compute(targets, predictions, probabilities)
    
    assert isinstance(res, EvaluationResult)
    assert res.accuracy == 5.0 / 6.0
    
    # Recall per class: 0: 2/2, 1: 2/2, 2: 1/2
    assert np.allclose(res.recall_per_class, [1.0, 1.0, 0.5])
    
    # Precision per class: 0: 2/2, 1: 2/3, 2: 1/1
    assert np.allclose(res.precision_per_class, [1.0, 2.0/3.0, 1.0])
    
    assert res.confusion_matrix.shape == (3, 3)
    assert res.normalized_confusion_matrix.shape == (3, 3)
    
    # Check normalization: row sums should be 1
    row_sums = res.normalized_confusion_matrix.sum(axis=1)
    assert np.allclose(row_sums, [1.0, 1.0, 1.0])


def test_empty_dataset_handling(mock_dataset_and_config, tmp_path):
    """Test rejecting empty evaluation dataset."""
    _, config, _ = mock_dataset_and_config
    
    # Create an empty dataset
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with open(empty_dir / "manifest.json", "w") as f:
        json.dump([], f)
        
    empty_dataset = SequenceGraphDataset(str(empty_dir))
    engine = EvaluationEngine(config, torch.device("cpu"))
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        engine.evaluate(empty_dataset)


def test_metrics_computer_empty_targets():
    """Test rejecting empty targets in metrics computer."""
    computer = MetricsComputer(num_classes=3)
    with pytest.raises(ValueError, match="Empty targets"):
        computer.compute(np.array([]), np.array([]), np.array([]))


def test_checkpoint_loading(mock_dataset_and_config):
    """Test loading a mock checkpoint through the CheckpointManager."""
    dataset, config, tmp_dir = mock_dataset_and_config
    engine = EvaluationEngine(config, torch.device("cpu"))
    
    # Generate and save a dummy checkpoint
    ckpt_dir = Path(tmp_dir) / "checkpoints"
    manager = CheckpointManager(str(ckpt_dir))
    
    manager.save(
        model=engine.model,
        optimizer=torch.optim.SGD(engine.model.parameters(), lr=0.1),
        scheduler=None,
        epoch=1,
        best_val_metric=1.0,
        config=config,
    )
    
    latest_path = str(ckpt_dir / "latest.pt")
    
    # Make a tiny mutation to model weights to ensure loading overwrites it
    with torch.no_grad():
        for param in engine.model.parameters():
            param.add_(1.0)
            
    engine.load_checkpoint(latest_path)
    
    # Verify weight restoration (the checkpoint saved them prior to addition)
    loaded_state = torch.load(latest_path, weights_only=False)
    for name, param in engine.model.named_parameters():
        assert torch.allclose(param, loaded_state["model_state"][name])


def test_evaluation_engine_deterministic_output(mock_dataset_and_config):
    """Test evaluate() on the same dataset/model produces identical results."""
    dataset, config, _ = mock_dataset_and_config
    
    engine = EvaluationEngine(config, torch.device("cpu"))
    # Fix the model weights
    torch.manual_seed(42)
    for param in engine.model.parameters():
        torch.nn.init.uniform_(param)
        
    res1 = engine.evaluate(dataset)
    res2 = engine.evaluate(dataset)
    
    assert np.array_equal(res1.predictions, res2.predictions)
    assert np.allclose(res1.probabilities, res2.probabilities)
    assert res1.accuracy == res2.accuracy


def test_shape_violations(mock_dataset_and_config, monkeypatch):
    """Test that shape violations are caught inside evaluate()."""
    dataset, config, _ = mock_dataset_and_config
    engine = EvaluationEngine(config, torch.device("cpu"))
    
    original_model = engine.model
    
    # Mock model to return wrong probability shape
    class MockModel(torch.nn.Module):
        def forward(self, x):
            # return shape (batch, 2) instead of (batch, 3)
            return torch.zeros((len(x), 2))
            
        def parameters(self):
            return original_model.parameters()
            
    engine.model = MockModel()
    
    with pytest.raises(ValueError, match="Shape violation: Expected probabilities"):
        engine.evaluate(dataset)
