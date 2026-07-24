"""
Tests for training/validate_dataset.py
"""

import json

import pytest
from training.validate_dataset import DatasetValidator


@pytest.fixture
def mock_dataset(tmp_path):
    data_dir = tmp_path / "simulated"
    data_dir.mkdir()
    
    # Manifest
    manifest = {
        "records": [
            {"sequence_id": "valid_seq"},
            {"sequence_id": "empty_seq"},
            {"sequence_id": "invalid_dim_seq"}
        ]
    }
    with open(data_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    # Valid Sequence
    valid = {
        "sequence_id": "valid_seq",
        "positions": [[[0,0], [1,1]], [[0,0], [2,2]]],
        "velocities": [[[0.1,0], [0,0.1]], [[0.1,0], [0,0.1]]],
        "frame_labels": ["Safe", "Safe"]
    }
    with open(data_dir / "valid_seq.json", "w") as f:
        json.dump(valid, f)
        
    # Empty Sequence (0 nodes)
    empty = {
        "sequence_id": "empty_seq",
        "positions": [[], []],
        "velocities": [[], []],
        "frame_labels": ["Safe", "Safe"]
    }
    with open(data_dir / "empty_seq.json", "w") as f:
        json.dump(empty, f)
        
    # Invalid Dimensions (missing coordinates)
    invalid = {
        "sequence_id": "invalid_dim_seq",
        "positions": [[[0], [1]]],  # 1D instead of 2D
        "velocities": [[[0.1], [0.1]]],
        "frame_labels": ["Safe"]
    }
    with open(data_dir / "invalid_dim_seq.json", "w") as f:
        json.dump(invalid, f)
        
    return data_dir


def test_validator_success(mock_dataset, tmp_path):
    out_dir = tmp_path / "out"
    validator = DatasetValidator(mock_dataset, out_dir)
    stats = validator.validate()
    
    assert stats.total_sequences == 3
    assert stats.total_frames == 5
    assert stats.empty_graph_count == 2
    assert stats.invalid_graph_count == 1
    assert stats.class_distribution == {"Safe": 5}  # 2 in valid, 2 in empty, 1 in invalid
    
    # Check outputs generated
    assert (out_dir / "dataset_statistics.json").exists()
    assert (out_dir / "dataset_summary.md").exists()
    assert (out_dir / "class_distribution.csv").exists()


def test_missing_manifest(tmp_path):
    out_dir = tmp_path / "out"
    data_dir = tmp_path / "empty_dir"
    data_dir.mkdir()
    
    validator = DatasetValidator(data_dir, out_dir)
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        validator.validate()


def test_malformed_json_sequence(tmp_path):
    out_dir = tmp_path / "out"
    data_dir = tmp_path / "bad_json"
    data_dir.mkdir()
    
    with open(data_dir / "manifest.json", "w") as f:
        json.dump({"records": [{"sequence_id": "bad"}]}, f)
        
    with open(data_dir / "bad.json", "w") as f:
        f.write("{ bad json ")
        
    validator = DatasetValidator(data_dir, out_dir)
    stats = validator.validate()
    
    assert stats.total_sequences == 0
    assert stats.invalid_graph_count == 1


def test_missing_sequence_file(tmp_path):
    out_dir = tmp_path / "out"
    data_dir = tmp_path / "missing"
    data_dir.mkdir()
    
    with open(data_dir / "manifest.json", "w") as f:
        json.dump({"records": [{"sequence_id": "missing_seq"}]}, f)
        
    validator = DatasetValidator(data_dir, out_dir)
    with pytest.raises(FileNotFoundError, match="Missing sequence file"):
        validator.validate()
