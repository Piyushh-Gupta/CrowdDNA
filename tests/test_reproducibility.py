import os
from training.reproducibility.builder import ManifestBuilder
from training.reproducibility.hashing import compute_deterministic_hash
from training.reproducibility.fingerprint import compute_dataset_fingerprint
from training.reproducibility.validator import ReproducibilityValidator
from training.reproducibility.diff import ManifestDiffer
from training.reproducibility.manifest import ManifestManager
from training.reproducibility.migration import SchemaMigrator

def test_deterministic_hashing():
    d1 = {"a": 1, "b": [1, 2, 3], "c": {"x": 10}}
    d2 = {"c": {"x": 10}, "b": [1, 2, 3], "a": 1}
    assert compute_deterministic_hash(d1) == compute_deterministic_hash(d2)

def test_dataset_fingerprint(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "file1.txt").write_text("hello")
    (d / "file2.txt").write_text("world")
    
    fast_fp = compute_dataset_fingerprint(str(d), strict=False)
    strict_fp = compute_dataset_fingerprint(str(d), strict=True)
    
    assert fast_fp != strict_fp
    assert len(fast_fp) == 64
    assert len(strict_fp) == 64

def test_builder():
    builder = ManifestBuilder(
        cli_invocation="python train.py",
        dataset_path="invalid_path",
        configuration={"epochs": 10}
    )
    builder.add_parent("parent_123")
    builder.add_artifact("model", "hash_456")
    manifest = builder.build()
    
    assert manifest.manifest_version == SchemaMigrator.CURRENT_VERSION
    assert manifest.crowddna_schema == "v1.0.0"
    assert manifest.configuration_hash == compute_deterministic_hash({"epochs": 10})
    assert "parent_123" in manifest.lineage_parents
    assert manifest.artifact_provenance["model"] == "hash_456"

def test_validator_and_registry():
    # We test the scoring logic using built-in validators
    builder = ManifestBuilder("python train.py", "invalid", {})
    manifest = builder.build()
    
    # Intentionally modify the environment snapshot to force a validation failure
    # Git commit validator is CRITICAL (100 deduction)
    from dataclasses import replace
    bad_env = replace(manifest.environment, git_commit="drifted")
    
    validator = ReproducibilityValidator()
    score, diagnostics = validator.validate(manifest, bad_env)
    
    assert score == 0.0 # 100 - 100 = 0
    assert any(d['validator'] == 'git_commit_match' and d['status'] == 'FAILED' for d in diagnostics)

def test_diff():
    builder1 = ManifestBuilder("cmd", "invalid", {"a": 1})
    m1 = builder1.build()
    builder2 = ManifestBuilder("cmd", "invalid", {"a": 2})
    m2 = builder2.build()
    
    diff_str = ManifestDiffer.diff_manifests(m1, m2)
    assert "~ configuration.a: 1 -> 2" in diff_str

def test_manifest_io(tmp_path):
    out_file = str(tmp_path / "manifest.json")
    builder = ManifestBuilder("cmd", "invalid", {"a": 1})
    manifest = builder.build()
    
    ManifestManager.save(manifest, out_file)
    assert os.path.exists(out_file)
    
    loaded = ManifestManager.load(out_file)
    assert loaded.manifest_hash == manifest.manifest_hash
    assert loaded.environment.python_version == manifest.environment.python_version

def test_migration():
    # Simulate loading an old 0.9 schema
    raw = {
        "manifest_version": "0.9",
        "cli_invocation": "old",
        "lineage_parents": [],
        "environment": {"pip_freeze": []}
    }
    migrated = SchemaMigrator.migrate(raw)
    assert migrated["manifest_version"] == "1.0"
    assert migrated["crowddna_schema"] == "v1.0.0"
    assert "validator_metadata" in migrated
