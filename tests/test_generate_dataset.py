"""
Smoke tests for generate_dataset() and _load_simulation_config()
in training/simulate_data.py.

Tests verify:
- _load_simulation_config() returns all required keys from a valid YAML.
- _load_simulation_config() raises FileNotFoundError for missing files.
- _load_simulation_config() raises KeyError when the simulation section is absent.
- _load_simulation_config() raises KeyError when a required key is missing.
- generate_dataset() creates one JSON file per (scenario × run).
- generate_dataset() creates manifest.json.
- generate_dataset() is deterministic (same files on two runs with same seed).
- generate_dataset() seeds are globally unique across scenarios.
"""

import json
import os
import pathlib
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from training.simulate_data import (
    _load_simulation_config,
    generate_dataset,
    RISK_CLASSES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(tmp_dir: str, num_runs: int = 2, num_timesteps: int = 5) -> str:
    """Writes a minimal valid YAML config to a temp directory."""
    cfg = {
        "simulation": {
            "num_runs_per_scenario": num_runs,
            "num_timesteps": num_timesteps,
            "random_seed_base": 0,
            "output_dir": os.path.join(tmp_dir, "output"),
        }
    }
    path = os.path.join(tmp_dir, "test_config.yaml")
    with open(path, "w") as fh:
        yaml.dump(cfg, fh)
    return path


# ---------------------------------------------------------------------------
# _load_simulation_config
# ---------------------------------------------------------------------------

class TestLoadSimulationConfig:
    def test_returns_all_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp)
            cfg = _load_simulation_config(path)
        required = {"num_runs_per_scenario", "num_timesteps", "random_seed_base", "output_dir"}
        assert required.issubset(cfg.keys())

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _load_simulation_config("does_not_exist.yaml")

    def test_raises_key_error_missing_simulation_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.yaml")
            with open(path, "w") as fh:
                yaml.dump({"other_section": {}}, fh)
            with pytest.raises(KeyError, match="simulation"):
                _load_simulation_config(path)

    def test_raises_key_error_on_missing_required_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.yaml")
            # Omit output_dir
            cfg = {
                "simulation": {
                    "num_runs_per_scenario": 2,
                    "num_timesteps": 5,
                    "random_seed_base": 0,
                    # output_dir missing
                }
            }
            with open(path, "w") as fh:
                yaml.dump(cfg, fh)
            with pytest.raises(KeyError):
                _load_simulation_config(path)

    def test_correct_values_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, num_runs=7, num_timesteps=15)
            cfg = _load_simulation_config(path)
        assert cfg["num_runs_per_scenario"] == 7
        assert cfg["num_timesteps"] == 15
        assert cfg["random_seed_base"] == 0


# ---------------------------------------------------------------------------
# generate_dataset (integration)
# ---------------------------------------------------------------------------

class TestGenerateDataset:
    """Uses num_runs_per_scenario=2, num_timesteps=5 to keep tests fast."""

    def test_creates_one_json_per_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, num_runs=2, num_timesteps=5)
            cfg = _load_simulation_config(path)
            generate_dataset(path)
            json_files = [
                f for f in os.listdir(cfg["output_dir"])
                if f.endswith(".json") and f != "manifest.json"
            ]
        # 3 scenarios × 2 runs = 6 files
        assert len(json_files) == 6

    def test_creates_manifest_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, num_runs=2, num_timesteps=5)
            cfg = _load_simulation_config(path)
            generate_dataset(path)
            assert os.path.isfile(os.path.join(cfg["output_dir"], "manifest.json"))

    def test_manifest_has_correct_entry_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, num_runs=2, num_timesteps=5)
            cfg = _load_simulation_config(path)
            generate_dataset(path)
            with open(os.path.join(cfg["output_dir"], "manifest.json")) as fh:
                manifest = json.load(fh)
        assert len(manifest) == 6

    def test_manifest_covers_all_risk_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, num_runs=1, num_timesteps=5)
            cfg = _load_simulation_config(path)
            generate_dataset(path)
            with open(os.path.join(cfg["output_dir"], "manifest.json")) as fh:
                manifest = json.load(fh)
        risk_classes_in_manifest = {e["risk_class"] for e in manifest}
        assert risk_classes_in_manifest == set(RISK_CLASSES)

    def test_each_json_has_correct_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, num_runs=1, num_timesteps=5)
            cfg = _load_simulation_config(path)
            generate_dataset(path)
            out = cfg["output_dir"]
            json_files = [f for f in os.listdir(out) if f.endswith(".json") and f != "manifest.json"]
            for fname in json_files:
                with open(os.path.join(out, fname)) as fh:
                    data = json.load(fh)
                assert "positions" in data
                assert "velocities" in data
                assert "frame_labels" in data
                assert "label_distribution" in data
                assert "generation_seed" in data

    def test_deterministic_output(self):
        """Two runs with the same seed base must produce byte-identical JSON files."""
        with tempfile.TemporaryDirectory() as tmp1:
            with tempfile.TemporaryDirectory() as tmp2:
                p1 = _write_config(tmp1, num_runs=1, num_timesteps=3)
                p2 = _write_config(tmp2, num_runs=1, num_timesteps=3)
                cfg1 = _load_simulation_config(p1)
                cfg2 = _load_simulation_config(p2)
                generate_dataset(p1)
                generate_dataset(p2)
                files1 = sorted(
                    f for f in os.listdir(cfg1["output_dir"])
                    if f.endswith(".json") and f != "manifest.json"
                )
                files2 = sorted(
                    f for f in os.listdir(cfg2["output_dir"])
                    if f.endswith(".json") and f != "manifest.json"
                )
                assert files1 == files2
                for fname in files1:
                    content1 = open(os.path.join(cfg1["output_dir"], fname)).read()
                    content2 = open(os.path.join(cfg2["output_dir"], fname)).read()
                    assert content1 == content2, f"Non-deterministic output for {fname}"

    def test_seeds_are_unique_across_all_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, num_runs=2, num_timesteps=3)
            cfg = _load_simulation_config(path)
            generate_dataset(path)
            out = cfg["output_dir"]
            seeds = []
            for fname in os.listdir(out):
                if fname.endswith(".json") and fname != "manifest.json":
                    with open(os.path.join(out, fname)) as fh:
                        seeds.append(json.load(fh)["generation_seed"])
        assert len(seeds) == len(set(seeds)), "Duplicate seeds detected across runs"

    def test_raises_file_not_found_for_bad_config(self):
        with pytest.raises(FileNotFoundError):
            generate_dataset("nonexistent_config.yaml")

    def test_num_timesteps_in_output_matches_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, num_runs=1, num_timesteps=7)
            cfg = _load_simulation_config(path)
            generate_dataset(path)
            out = cfg["output_dir"]
            json_files = [f for f in os.listdir(out) if f.endswith(".json") and f != "manifest.json"]
            with open(os.path.join(out, json_files[0])) as fh:
                data = json.load(fh)
        assert data["num_timesteps"] == 7
        assert len(data["frame_labels"]) == 7
        assert len(data["positions"]) == 7
