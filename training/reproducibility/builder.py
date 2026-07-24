import uuid
from typing import Dict, Any, List
from training.reproducibility.metadata import ExperimentManifest
from training.reproducibility.environment import capture_environment_snapshot
from training.reproducibility.fingerprint import compute_dataset_fingerprint
from training.reproducibility.hashing import compute_deterministic_hash
from training.reproducibility.lineage import LineageManager
from training.reproducibility.migration import SchemaMigrator

class ManifestBuilder:
    """Orchestration layer for constructing an ExperimentManifest."""
    
    def __init__(self, cli_invocation: str, dataset_path: str, configuration: Dict[str, Any]):
        self.cli_invocation = cli_invocation
        self.dataset_path = dataset_path
        self.configuration = configuration
        self.parents: List[str] = []
        self.artifact_provenance: Dict[str, str] = {}
        
    def add_parent(self, parent_uuid_or_hash: str):
        """Adds a parent lineage reference."""
        self.parents.append(parent_uuid_or_hash)
        
    def add_artifact(self, name: str, artifact_hash: str):
        """Records an artifact hash for provenance."""
        self.artifact_provenance[name] = artifact_hash
        
    def build(self, strict_fingerprint: bool = False) -> ExperimentManifest:
        """Constructs the final manifest."""
        environment_snapshot = capture_environment_snapshot()
        fast_fingerprint = compute_dataset_fingerprint(self.dataset_path, strict=False)
        strict_fingerprint_hash = compute_dataset_fingerprint(self.dataset_path, strict=True) if strict_fingerprint else "not_computed"
        
        config_hash = compute_deterministic_hash(self.configuration)
        
        lineage_parents = LineageManager.construct_lineage(self.parents)
        
        manifest_uuid = str(uuid.uuid4())
        
        # We need to compute the manifest hash. We can hash a dict representation of the core fields.
        core_data = {
            "cli_invocation": self.cli_invocation,
            "fast_dataset_fingerprint": fast_fingerprint,
            "strict_dataset_fingerprint": strict_fingerprint_hash,
            "configuration_hash": config_hash,
            "lineage_parents": list(lineage_parents),
            "artifact_provenance": self.artifact_provenance,
            # We include some stable environment variables in the hash to ensure the exact environment is tracked
            "git_commit": environment_snapshot.git_commit,
            "git_is_dirty": environment_snapshot.git_is_dirty,
        }
        manifest_hash = compute_deterministic_hash(core_data)
        
        return ExperimentManifest(
            manifest_uuid=manifest_uuid,
            manifest_hash=manifest_hash,
            manifest_version=SchemaMigrator.CURRENT_VERSION,
            crowddna_schema="v1.0.0",
            cli_invocation=self.cli_invocation,
            environment=environment_snapshot,
            fast_dataset_fingerprint=fast_fingerprint,
            strict_dataset_fingerprint=strict_fingerprint_hash,
            configuration_hash=config_hash,
            configuration=self.configuration,
            lineage_parents=lineage_parents,
            artifact_provenance=dict(self.artifact_provenance)
        )
