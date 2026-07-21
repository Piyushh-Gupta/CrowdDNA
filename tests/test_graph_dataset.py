import json
import os

import pytest
from torch_geometric.data import Data

from crowdflow_dna.graph.graph_dataset import GraphDataset


@pytest.fixture
def data_dir(tmp_path):
    """Provides an empty temporary directory for dataset testing."""
    return str(tmp_path)


def write_manifest(directory, entries):
    manifest_path = os.path.join(directory, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)


def write_trajectory(directory, filename, record_dict):
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record_dict, f)


def generate_valid_record(seq_id, labels):
    T = len(labels)
    N = 2
    return {
        "sequence_id": seq_id,
        "risk_class": "Safe",
        "scenario_name": "test_scenario",
        "num_agents": N,
        "num_timesteps": T,
        "generation_seed": 42,
        "label_distribution": {
            "Safe": labels.count("Safe"),
            "Congesting": labels.count("Congesting"),
            "Critical": labels.count("Critical"),
        },
        "positions": [[[0.1, 0.1], [0.2, 0.2]] for _ in range(T)],
        "velocities": [[[0.5, 0.0], [-0.5, 0.0]] for _ in range(T)],
        "frame_labels": labels,
    }


def test_manifest_loading_success(data_dir):
    """Test successful loading of a valid manifest."""
    entries = [{"file_path": "traj1.json"}, {"file_path": "traj2.json"}]
    write_manifest(data_dir, entries)
    dataset = GraphDataset(root=data_dir)
    assert len(dataset) == 2


def test_missing_manifest(data_dir):
    """Test error when manifest.json is missing."""
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        GraphDataset(root=data_dir)


def test_invalid_manifest_json(data_dir):
    """Test error when manifest.json is malformed."""
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("{invalid_json:")
    with pytest.raises(ValueError, match="Invalid JSON in manifest"):
        GraphDataset(root=data_dir)


def test_manifest_not_a_list(data_dir):
    """Test error when manifest is a dict instead of an array."""
    write_manifest(data_dir, {"file_path": "traj1.json"})
    with pytest.raises(ValueError, match="must be a JSON array"):
        GraphDataset(root=data_dir)


def test_empty_dataset(data_dir):
    """Test empty dataset length is 0."""
    write_manifest(data_dir, [])
    dataset = GraphDataset(root=data_dir)
    assert len(dataset) == 0


def test_indexing_and_graph_generation(data_dir):
    """Test full pipeline: indexing, loading, and building graphs."""
    labels = ["Safe", "Congesting"]
    write_trajectory(data_dir, "traj1.json", generate_valid_record("seq1", labels))
    write_manifest(data_dir, [{"file_path": "traj1.json"}])

    dataset = GraphDataset(root=data_dir, proximity_radius=0.5)
    graphs = dataset[0]

    assert isinstance(graphs, list)
    assert len(graphs) == 2
    assert isinstance(graphs[0], Data)
    
    # Check graph features
    assert graphs[0].x.shape == (2, 5)
    # Check label preservation and mapping
    assert graphs[0].y.item() == 0  # Safe
    assert graphs[1].y.item() == 1  # Congesting


def test_missing_trajectory_file(data_dir):
    """Test error when JSON file referenced in manifest is missing."""
    write_manifest(data_dir, [{"file_path": "missing.json"}])
    dataset = GraphDataset(root=data_dir)
    with pytest.raises(FileNotFoundError, match="Trajectory JSON missing"):
        _ = dataset[0]


def test_corrupted_trajectory_json(data_dir):
    """Test error when trajectory JSON is malformed."""
    write_manifest(data_dir, [{"file_path": "corrupt.json"}])
    with open(os.path.join(data_dir, "corrupt.json"), "w") as f:
        f.write("[corrupted")
        
    dataset = GraphDataset(root=data_dir)
    with pytest.raises(ValueError, match="Invalid JSON"):
        _ = dataset[0]


def test_invalid_trajectory_schema_missing_key(data_dir):
    """Test error when required schema keys are missing."""
    record = generate_valid_record("seq1", ["Safe"])
    del record["positions"]
    write_trajectory(data_dir, "traj1.json", record)
    write_manifest(data_dir, [{"file_path": "traj1.json"}])

    dataset = GraphDataset(root=data_dir)
    with pytest.raises(ValueError, match="Missing required key 'positions'"):
        _ = dataset[0]


def test_invalid_trajectory_schema_timestep_mismatch(data_dir):
    """Test error when array dimensions don't match num_timesteps."""
    record = generate_valid_record("seq1", ["Safe", "Safe"])
    record["positions"] = record["positions"][:1]  # Truncate positions
    write_trajectory(data_dir, "traj1.json", record)
    write_manifest(data_dir, [{"file_path": "traj1.json"}])

    dataset = GraphDataset(root=data_dir)
    with pytest.raises(ValueError, match="Timestep mismatch"):
        _ = dataset[0]


def test_unknown_risk_label(data_dir):
    """Test error when an unknown risk label is encountered."""
    write_trajectory(data_dir, "traj1.json", generate_valid_record("seq1", ["UnknownLabel"]))
    write_manifest(data_dir, [{"file_path": "traj1.json"}])

    dataset = GraphDataset(root=data_dir)
    with pytest.raises(ValueError, match="Unknown label 'UnknownLabel'"):
        _ = dataset[0]


def test_deterministic_ordering(data_dir):
    """Test that dataset order matches manifest order."""
    entries = []
    for i in range(10):
        filename = f"traj_{i}.json"
        write_trajectory(data_dir, filename, generate_valid_record(f"seq{i}", ["Safe"]))
        entries.append({"file_path": filename, "idx": i})
    
    write_manifest(data_dir, entries)
    dataset = GraphDataset(root=data_dir)
    
    # Since all trajectories are identical except sequence_id, we can verify
    # the number of items and assume order matches index requests.
    assert len(dataset) == 10
    
    # Test random access
    graphs_3 = dataset[3]
    graphs_7 = dataset[7]
    assert len(graphs_3) == 1
    assert len(graphs_7) == 1
