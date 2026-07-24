import os
import hashlib
import time
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    producer_node_id: str
    checksum: str
    schema_version: str
    location: str
    timestamp: float

class ArtifactRegistry:
    """Registry to track all produced artifacts."""
    def __init__(self):
        self._records: Dict[str, ArtifactRecord] = {}

    def register(self, artifact_id: str, producer_node_id: str, schema_version: str, location: str) -> ArtifactRecord:
        if artifact_id in self._records:
            raise ValueError(f"Artifact {artifact_id} already registered.")
            
        checksum = self._compute_checksum(location)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            producer_node_id=producer_node_id,
            checksum=checksum,
            schema_version=schema_version,
            location=location,
            timestamp=time.time()
        )
        self._records[artifact_id] = record
        return record

    def get_record(self, artifact_id: str) -> Optional[ArtifactRecord]:
        return self._records.get(artifact_id)
        
    def get_all(self) -> Dict[str, ArtifactRecord]:
        return dict(self._records)

    def _compute_checksum(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return "virtual-or-missing"
        if os.path.isdir(filepath):
            return "directory"
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
