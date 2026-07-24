import json
import os
import tempfile
from typing import Dict, Any
from training.orchestration.metadata import ExecutionPlan, ExecutionCursor, WorkflowExecution
from training.orchestration.artifacts import ArtifactRegistry

class CheckpointManager:
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        self.checkpoint_file = os.path.join(self.directory, "checkpoint.json")

    def save(self, execution: WorkflowExecution, plan: ExecutionPlan, cursor: ExecutionCursor, artifact_registry: ArtifactRegistry) -> None:
        data = {
            "execution": {
                "execution_id": execution.execution_id,
                "session_id": execution.session_id,
                "artifact_tree_root": execution.artifact_tree_root
            },
            "cursor": {
                "node_states": {k: v.name for k, v in cursor.node_states.items()},
                "retry_counts": cursor.retry_counts,
                "completed_nodes": sorted(list(cursor.completed_nodes)),
                "failed_nodes": sorted(list(cursor.failed_nodes)),
                "skipped_nodes": sorted(list(cursor.skipped_nodes))
            },
            "artifacts": [
                {
                    "artifact_id": rec.artifact_id,
                    "producer_node_id": rec.producer_node_id,
                    "checksum": rec.checksum,
                    "schema_version": rec.schema_version,
                    "location": rec.location,
                    "timestamp": rec.timestamp
                } for artifact_id, rec in sorted(artifact_registry.get_all().items())
            ]

        }
        
        # Write to temp file and rename for atomicity
        fd, temp_path = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(temp_path, self.checkpoint_file)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.checkpoint_file):
            return {}
        with open(self.checkpoint_file, 'r') as f:
            return json.load(f)
