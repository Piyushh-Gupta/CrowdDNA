from dataclasses import dataclass, field
from typing import Mapping, Any

@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Immutable snapshot of the system and python environment."""
    os_name: str
    python_version: str
    pytorch_version: str
    cuda_version: str
    gpu_model: str
    gpu_driver_version: str
    vram_total_mb: float
    cpu_model: str
    ram_total_mb: float
    git_commit: str
    git_branch: str
    git_is_dirty: bool
    pip_freeze: tuple[str, ...]
    env_vars: Mapping[str, str]

@dataclass(frozen=True)
class ExperimentManifest:
    """Immutable manifest for an experiment run ensuring complete reproducibility."""
    manifest_uuid: str
    manifest_hash: str
    manifest_version: str
    crowddna_schema: str
    cli_invocation: str
    environment: EnvironmentSnapshot
    fast_dataset_fingerprint: str
    strict_dataset_fingerprint: str
    configuration_hash: str
    configuration: Mapping[str, Any]
    lineage_parents: tuple[str, ...]
    artifact_provenance: Mapping[str, str]
    validator_metadata: Mapping[str, Any] = field(default_factory=dict)
