import json
from dataclasses import dataclass, asdict

@dataclass
class ReleaseManifest:
    release_id: str
    release_version: str
    git_commit: str
    image_digest: str
    build_timestamp: str
    frontend_version: str
    api_version: str
    framework_version: str
    reproducibility_manifest_reference: str
    deployment_manifest_reference: str
    artifact_manifest_reference: str

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)
        
    @classmethod
    def from_json(cls, data: str):
        return cls(**json.loads(data))

@dataclass
class BackupManifest:
    backup_id: str
    release_id: str
    git_commit: str
    created_at: str
    sha256: str
    compressed: bool
    verified: bool
    manifest_version: str
    backup_size: int

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)
        
    @classmethod
    def from_json(cls, data: str):
        return cls(**json.loads(data))

@dataclass
class ArtifactManifest:
    artifact_id: str
    checksum: str
    location: str

@dataclass
class DeploymentManifest:
    deployment_id: str
    release_id: str
    target_environment: str
    status: str

@dataclass
class VersionCompatibility:
    backend_version: str
    frontend_version: str
    is_compatible: bool