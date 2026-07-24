# Experiment Reproducibility & Versioning Framework

The Phase 17 reproducibility framework is designed to capture, validate, and compare complete CrowdDNA experiment environments. It ensures that deterministic lineage and full environmental states are serialized safely for every run.

## Architecture

The framework relies on the following core components:
- **`metadata.py`**: Immutable models representing the state (`ExperimentManifest`, `EnvironmentSnapshot`).
- **`builder.py`**: Orchestrates environment hashing, dataset fingerprinting, and metadata capturing to compile the final manifest.
- **`manifest.py`**: Manages stable disk IO of schemas.
- **`migration.py`**: Ensures schemas can gracefully upgrade over time.
- **`fingerprint.py`**: Performs two-tier hashing on massive datasets to bind experimental results to semantic data versions securely.
- **`validator.py`**: Executes severity-weighted validation against environments via `registry.py` rules.
- **`diff.py`**: Formats differences into a human-readable structure to visually pinpoint dataset, config, or CUDA drift.

## Manifests

Every experiment run builds an `ExperimentManifest` (serialized as `reproducibility_manifest.json`), which serves as the ultimate source of truth regarding how a training artifact (`deployment.pt`) was built. It acts as an audit trail.

## Fingerprints

- **Fast Fingerprint**: Used operationally. Only checks filenames, modification dates, and sizes.
- **Strict Fingerprint**: Cryptographic deep hash (SHA-256) of all file contents (slower, extremely rigorous).

## CLI Usage

The package exposes standard interactions:

1. **Snapshot**: `python -m training.snapshot_environment snapshot`
2. **Validate**: `python -m training.snapshot_environment validate <manifest_path>`
3. **Diff**: `python -m training.snapshot_environment diff <manifest1_path> <manifest2_path>`
