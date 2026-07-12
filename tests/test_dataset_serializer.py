"""
Smoke tests for DatasetSerializer in training/simulate_data.py.

Tests verify:
- save_record() writes a valid JSON file with the correct schema.
- save_record() calls validate() and raises ValueError on bad records.
- save_record() creates output directories on demand.
- save_record() returns the absolute path to the written file.
- write_manifest() writes manifest.json with correct entries.
- write_manifest() sorts entries by sequence_id deterministically.
- write_manifest() works with an empty record list.
- JSON output is deterministic (byte-identical on repeated calls).
"""

import json
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from training.simulate_data import (
    DatasetSerializer,
    RISK_CLASSES,
    TrajectoryRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    sequence_id: str = "safe_run_000",
    risk_class: str = "Safe",
    scenario_name: str = "safe_open_field",
    num_agents: int = 2,
    num_timesteps: int = 3,
    generation_seed: int = 42,
) -> TrajectoryRecord:
    """Builds a minimal valid TrajectoryRecord for testing."""
    positions = [[[1.0, 2.0], [3.0, 4.0]] for _ in range(num_timesteps)]
    velocities = [[[0.1, 0.2], [0.3, 0.4]] for _ in range(num_timesteps)]
    frame_labels = ["Safe"] * num_timesteps
    label_distribution = {cls: 0 for cls in RISK_CLASSES}
    label_distribution["Safe"] = num_timesteps
    return TrajectoryRecord(
        sequence_id=sequence_id,
        risk_class=risk_class,
        scenario_name=scenario_name,
        num_agents=num_agents,
        num_timesteps=num_timesteps,
        generation_seed=generation_seed,
        label_distribution=label_distribution,
        positions=positions,
        velocities=velocities,
        frame_labels=frame_labels,
    )


# ---------------------------------------------------------------------------
# save_record
# ---------------------------------------------------------------------------

class TestSaveRecord:
    def test_creates_file_with_correct_name(self):
        serializer = DatasetSerializer()
        record = _make_record(sequence_id="safe_run_001")
        with tempfile.TemporaryDirectory() as tmp:
            path = serializer.save_record(record, tmp)
            assert os.path.isfile(path)
            assert os.path.basename(path) == "safe_run_001.json"

    def test_returns_absolute_path(self):
        serializer = DatasetSerializer()
        record = _make_record()
        with tempfile.TemporaryDirectory() as tmp:
            path = serializer.save_record(record, tmp)
            assert os.path.isabs(path)

    def test_json_schema_matches_trajectory_record(self):
        serializer = DatasetSerializer()
        record = _make_record()
        with tempfile.TemporaryDirectory() as tmp:
            path = serializer.save_record(record, tmp)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        expected_keys = {
            "sequence_id", "risk_class", "scenario_name",
            "num_agents", "num_timesteps", "generation_seed",
            "label_distribution", "positions", "velocities", "frame_labels",
        }
        assert set(data.keys()) == expected_keys

    def test_json_values_match_record_fields(self):
        serializer = DatasetSerializer()
        record = _make_record(num_agents=2, num_timesteps=3, generation_seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            path = serializer.save_record(record, tmp)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        assert data["sequence_id"] == record.sequence_id
        assert data["risk_class"] == record.risk_class
        assert data["num_agents"] == record.num_agents
        assert data["num_timesteps"] == record.num_timesteps
        assert data["generation_seed"] == record.generation_seed
        assert len(data["positions"]) == record.num_timesteps
        assert len(data["velocities"]) == record.num_timesteps
        assert len(data["frame_labels"]) == record.num_timesteps

    def test_creates_output_directory_if_missing(self):
        serializer = DatasetSerializer()
        record = _make_record()
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c")
            assert not os.path.exists(nested)
            serializer.save_record(record, nested)
            assert os.path.isdir(nested)

    def test_raises_value_error_on_invalid_record(self):
        serializer = DatasetSerializer()
        record = _make_record()
        # Corrupt risk_class to trigger validate()
        record.risk_class = "INVALID"
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError):
                serializer.save_record(record, tmp)

    def test_file_not_created_when_validation_fails(self):
        serializer = DatasetSerializer()
        record = _make_record()
        record.risk_class = "INVALID"
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError):
                serializer.save_record(record, tmp)
            # Directory exists but no .json file should have been written
            json_files = list(pathlib.Path(tmp).glob("*.json"))
            assert len(json_files) == 0

    def test_output_is_deterministic(self):
        """Identical record must produce byte-identical files on two writes."""
        serializer = DatasetSerializer()
        record = _make_record()
        with tempfile.TemporaryDirectory() as tmp1:
            with tempfile.TemporaryDirectory() as tmp2:
                path1 = serializer.save_record(record, tmp1)
                path2 = serializer.save_record(record, tmp2)
                assert open(path1).read() == open(path2).read()

    def test_json_has_sorted_keys(self):
        serializer = DatasetSerializer()
        record = _make_record()
        with tempfile.TemporaryDirectory() as tmp:
            path = serializer.save_record(record, tmp)
            raw = open(path, encoding="utf-8").read()
        # sorted_keys=True means the first key in the file is the
        # lexicographically smallest; check that "frame_labels" precedes
        # "generation_seed" in the raw text.
        assert raw.index('"frame_labels"') < raw.index('"generation_seed"')

    def test_file_ends_with_newline(self):
        serializer = DatasetSerializer()
        record = _make_record()
        with tempfile.TemporaryDirectory() as tmp:
            path = serializer.save_record(record, tmp)
            raw = open(path, encoding="utf-8").read()
        assert raw.endswith("\n")


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

class TestWriteManifest:
    def test_creates_manifest_json(self):
        serializer = DatasetSerializer()
        record = _make_record()
        with tempfile.TemporaryDirectory() as tmp:
            serializer.write_manifest([record], tmp)
            assert os.path.isfile(os.path.join(tmp, "manifest.json"))

    def test_manifest_is_json_array(self):
        serializer = DatasetSerializer()
        record = _make_record()
        with tempfile.TemporaryDirectory() as tmp:
            serializer.write_manifest([record], tmp)
            with open(os.path.join(tmp, "manifest.json"), encoding="utf-8") as fh:
                data = json.load(fh)
        assert isinstance(data, list)

    def test_manifest_entry_fields(self):
        serializer = DatasetSerializer()
        record = _make_record()
        with tempfile.TemporaryDirectory() as tmp:
            serializer.write_manifest([record], tmp)
            with open(os.path.join(tmp, "manifest.json"), encoding="utf-8") as fh:
                data = json.load(fh)
        entry = data[0]
        expected_keys = {
            "sequence_id", "risk_class", "scenario_name",
            "num_agents", "num_timesteps", "generation_seed",
            "label_distribution", "file_path",
        }
        assert set(entry.keys()) == expected_keys

    def test_manifest_file_path_is_relative(self):
        serializer = DatasetSerializer()
        record = _make_record(sequence_id="safe_run_007")
        with tempfile.TemporaryDirectory() as tmp:
            serializer.write_manifest([record], tmp)
            with open(os.path.join(tmp, "manifest.json"), encoding="utf-8") as fh:
                data = json.load(fh)
        assert data[0]["file_path"] == "safe_run_007.json"

    def test_manifest_sorted_by_sequence_id(self):
        serializer = DatasetSerializer()
        records = [
            _make_record(sequence_id="safe_run_002"),
            _make_record(sequence_id="safe_run_001"),
            _make_record(sequence_id="safe_run_003"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            serializer.write_manifest(records, tmp)
            with open(os.path.join(tmp, "manifest.json"), encoding="utf-8") as fh:
                data = json.load(fh)
        ids = [e["sequence_id"] for e in data]
        assert ids == sorted(ids)

    def test_empty_record_list_writes_empty_array(self):
        serializer = DatasetSerializer()
        with tempfile.TemporaryDirectory() as tmp:
            serializer.write_manifest([], tmp)
            with open(os.path.join(tmp, "manifest.json"), encoding="utf-8") as fh:
                data = json.load(fh)
        assert data == []

    def test_manifest_creates_output_directory(self):
        serializer = DatasetSerializer()
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "x", "y")
            assert not os.path.exists(nested)
            serializer.write_manifest([], nested)
            assert os.path.isdir(nested)

    def test_manifest_count_matches_records(self):
        serializer = DatasetSerializer()
        records = [_make_record(sequence_id=f"safe_run_{i:03d}") for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            serializer.write_manifest(records, tmp)
            with open(os.path.join(tmp, "manifest.json"), encoding="utf-8") as fh:
                data = json.load(fh)
        assert len(data) == 5

    def test_manifest_is_deterministic(self):
        serializer = DatasetSerializer()
        records = [_make_record(sequence_id=f"run_{i:03d}") for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp1:
            with tempfile.TemporaryDirectory() as tmp2:
                serializer.write_manifest(records, tmp1)
                serializer.write_manifest(records, tmp2)
                raw1 = open(os.path.join(tmp1, "manifest.json")).read()
                raw2 = open(os.path.join(tmp2, "manifest.json")).read()
        assert raw1 == raw2

    def test_manifest_ends_with_newline(self):
        serializer = DatasetSerializer()
        with tempfile.TemporaryDirectory() as tmp:
            serializer.write_manifest([], tmp)
            raw = open(os.path.join(tmp, "manifest.json"), encoding="utf-8").read()
        assert raw.endswith("\n")
