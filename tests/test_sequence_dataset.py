import json

import logging

import pytest
import torch
from torch_geometric.data import Data

from crowdflow_dna.graph.sequence_dataset import SequenceGraphDataset, SequenceSample


@pytest.fixture
def mock_trajectory_data(tmp_path):
    """Creates a temporary manifest and synthetic trajectory JSONs for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # 1. Valid trajectory (Risk: Safe)
    traj_1 = {
        "sequence_id": "seq_1",
        "risk_class": "Safe",
        "positions": [
            [[0.0, 0.0], [1.0, 1.0]],  # t=0
            [[0.1, 0.1], [1.1, 1.1]]   # t=1
        ],
        "velocities": [
            [[0.1, 0.1], [0.1, 0.1]],
            [[0.1, 0.1], [0.1, 0.1]]
        ],
        "frame_labels": ["Safe", "Safe"]
    }
    
    # 2. Valid trajectory (Risk: Critical)
    traj_2 = {
        "sequence_id": "seq_2",
        "risk_class": "Critical",
        "positions": [
            [[5.0, 5.0], [5.1, 5.1]],  # t=0
            [[5.2, 5.2], [5.3, 5.3]]   # t=1
        ],
        "velocities": [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]]
        ],
        "frame_labels": ["Congesting", "Critical"]
    }
    
    # 3. Missing risk_class, testing fallback
    traj_3 = {
        "sequence_id": "seq_3",
        "positions": [
            [[0.0, 0.0]],
            [[0.0, 0.0]]
        ],
        "velocities": [
            [[0.0, 0.0]],
            [[0.0, 0.0]]
        ],
        "frame_labels": ["Safe", "Congesting"]
    }
    
    # 4. Malformed trajectory (mismatched timesteps)
    traj_4 = {
        "sequence_id": "seq_4",
        "positions": [
            [[0.0, 0.0]]
        ],
        "velocities": [
            [[0.0, 0.0]],
            [[0.0, 0.0]]
        ],
        "frame_labels": ["Safe", "Safe"]
    }
    
    files = ["traj_1.json", "traj_2.json", "traj_3.json", "traj_4.json"]
    trajs = [traj_1, traj_2, traj_3, traj_4]
    
    manifest = []
    for fname, data in zip(files, trajs):
        fpath = data_dir / fname
        with open(fpath, "w") as f:
            json.dump(data, f)
        manifest.append({
            "file_path": fname,
            "num_timesteps": len(data["frame_labels"])
        })
        
    # Also add a missing file entry
    manifest.append({
        "file_path": "missing.json",
        "num_timesteps": 2
    })
    
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
        
    return str(data_dir)


def test_dataset_length(mock_trajectory_data):
    """Test that dataset length matches the number of sequences in the manifest."""
    dataset = SequenceGraphDataset(mock_trajectory_data)
    assert len(dataset) == 5  # 4 files + 1 missing


def test_trajectory_loading_and_graph_creation(mock_trajectory_data):
    """Test that get() returns a complete trajectory with correct PyG Data objects."""
    dataset = SequenceGraphDataset(mock_trajectory_data, proximity_radius=2.0)
    
    sample = dataset[0]  # traj_1
    assert isinstance(sample, SequenceSample)
    assert sample.sequence_id == "seq_1"
    
    graphs = sample.graphs
    assert len(graphs) == 2  # 2 timesteps
    
    for data in graphs:
        assert isinstance(data, Data)
        # metadata is no longer attached directly to Data objects
        assert not hasattr(data, "sequence_id")
        assert not hasattr(data, "timestep")
        assert data.x.shape == (2, 5)  # 2 agents, 5 features (dx, dy, x, y, speed)
        # In traj 1, the agents are at [0,0] and [1,1]. Distance is sqrt(2) ~ 1.414 < 2.0
        # GraphBuilder connects neighbors without self-loops by default, yielding 2 directed edges.
        assert data.edge_index.shape[1] == 2


def test_sequence_labels(mock_trajectory_data):
    """Test that the sequence label is correctly assigned from risk_class and is a scalar."""
    dataset = SequenceGraphDataset(mock_trajectory_data)
    
    # traj_1 has risk_class = "Safe" (0)
    sample_1 = dataset[0]
    assert sample_1.label.item() == 0
    assert sample_1.label.shape == torch.Size([])  # verify it's a scalar
    
    # traj_2 has risk_class = "Critical" (2)
    sample_2 = dataset[1]
    assert sample_2.label.item() == 2


def test_fallback_sequence_label(mock_trajectory_data):
    """Test that fallback to max frame label works if risk_class is missing."""
    dataset = SequenceGraphDataset(mock_trajectory_data)
    
    # traj_3 has no risk_class, frame labels are "Safe" (0) and "Congesting" (1)
    # The max is 1.
    sample_3 = dataset[2]
    assert sample_3.label.item() == 1


def test_malformed_trajectory(mock_trajectory_data):
    """Test that ValueError is raised for inconsistent sequence dimensions."""
    dataset = SequenceGraphDataset(mock_trajectory_data)
    
    with pytest.raises(ValueError, match="Timestep mismatch"):
        _ = dataset[3]


def test_missing_file(mock_trajectory_data):
    """Test that FileNotFoundError is raised if a file in the manifest is missing."""
    dataset = SequenceGraphDataset(mock_trajectory_data)
    
    with pytest.raises(FileNotFoundError, match="Trajectory JSON missing"):
        _ = dataset[4]


def test_deterministic_ordering(mock_trajectory_data):
    """Test that iterating over the dataset is deterministic."""
    dataset = SequenceGraphDataset(mock_trajectory_data)
    
    # Extract sequence labels from the first three valid sequences
    labels = [dataset[i].label.item() for i in range(3)]
    assert labels == [0, 2, 1]


def test_lazy_loading(tmp_path):
    """Test that files are only loaded when get() is called."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    manifest = [{"file_path": "does_not_exist.json", "num_timesteps": 10}]
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
        
    # Initialization should succeed without raising FileNotFoundError
    dataset = SequenceGraphDataset(str(data_dir))
    assert len(dataset) == 1
    
    # The error should only be raised upon access
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_repeated_access_consistency(mock_trajectory_data):
    """Test that repeated access yields identical data structures."""
    dataset = SequenceGraphDataset(mock_trajectory_data)
    
    sample_a = dataset[0]
    sample_b = dataset[0]
    
    assert sample_a.label.item() == sample_b.label.item()
    assert len(sample_a.graphs) == len(sample_b.graphs)
    
    for ga, gb in zip(sample_a.graphs, sample_b.graphs):
        assert torch.allclose(ga.x, gb.x)
        assert torch.allclose(ga.edge_index, gb.edge_index)


def test_missing_sequence_id_warning(tmp_path, caplog):
    """Test that a warning is emitted if sequence_id is missing and a default is synthesized."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Trajectory missing sequence_id
    traj = {
        "risk_class": "Safe",
        "positions": [[[0.0, 0.0]]],
        "velocities": [[[0.0, 0.0]]],
        "frame_labels": ["Safe"]
    }
    
    with open(data_dir / "traj_missing_id.json", "w") as f:
        json.dump(traj, f)
        
    manifest = [{"file_path": "traj_missing_id.json", "num_timesteps": 1}]
    with open(data_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    dataset = SequenceGraphDataset(str(data_dir))
    
    with caplog.at_level(logging.WARNING):
        sample = dataset[0]
        
    assert "has no sequence_id; using default seq_0" in caplog.text
    assert sample.sequence_id == "seq_0"
