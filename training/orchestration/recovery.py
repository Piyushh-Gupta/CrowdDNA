from typing import Tuple
from training.orchestration.metadata import WorkflowManifest, ExecutionPlan, ExecutionCursor
from training.orchestration.artifacts import ArtifactRegistry, ArtifactRecord
from training.orchestration.checkpoint import CheckpointManager
from training.orchestration.exceptions import RecoveryFailure, CheckpointCorruptionError
from training.orchestration.state import NodeState
from training.orchestration.planner import WorkflowPlanner
from training.orchestration.validation import WorkflowValidator

class RecoveryEngine:
    @staticmethod
    def resume(manifest: WorkflowManifest, checkpoint_dir: str) -> Tuple[ExecutionPlan, ExecutionCursor, ArtifactRegistry]:
        # 1. Validate manifest DAG
        WorkflowValidator.validate(manifest.definition)
        
        # 2. Build Plan
        plan, fresh_cursor = WorkflowPlanner.plan(manifest.definition)
        
        # 3. Load Checkpoint
        manager = CheckpointManager(checkpoint_dir)
        data = manager.load()
        if not data:
            raise RecoveryFailure("No checkpoint found to resume from.")
            
        # 4. Verify execution IDs match
        stored_exec = data.get("execution", {})
        if stored_exec.get("execution_id") != manifest.execution.execution_id:
            raise CheckpointCorruptionError("Checkpoint execution ID does not match the manifest.")
            
        # 5. Restore Artifact Registry
        registry = ArtifactRegistry()
        for art in data.get("artifacts", []):
            # Bypass direct register to restore exact record
            record = ArtifactRecord(
                artifact_id=art["artifact_id"],
                producer_node_id=art["producer_node_id"],
                checksum=art["checksum"],
                schema_version=art["schema_version"],
                location=art["location"],
                timestamp=art["timestamp"]
            )
            registry._records[record.artifact_id] = record
            
        # 6. Restore Cursor
        cursor = ExecutionCursor()
        stored_cursor = data.get("cursor", {})
        for node_id, state_str in stored_cursor.get("node_states", {}).items():
            if node_id not in manifest.definition.nodes:
                raise CheckpointCorruptionError(f"Checkpoint contains node '{node_id}' not found in manifest definition.")
            try:
                state = NodeState[state_str]
                # If node was running or compensating when crashed, reset it to pending
                # Or keep it as failed based on idempotent nature. Let's reset RUNNING -> PENDING
                if state in (NodeState.RUNNING, NodeState.COMPENSATING, NodeState.PREPARING):
                    state = NodeState.PENDING
                cursor.node_states[node_id] = state
            except KeyError:
                raise CheckpointCorruptionError(f"Unknown node state '{state_str}' in checkpoint.")
                
        cursor.retry_counts = stored_cursor.get("retry_counts", {})
        cursor.completed_nodes = set(stored_cursor.get("completed_nodes", []))
        cursor.failed_nodes = set(stored_cursor.get("failed_nodes", []))
        
        # Filter completed nodes that shouldn't be completed if we reset state
        # Actually update_state in cursor does the set management, but we just manually rebuilt it.
        # Let's cleanly sync the sets
        cursor.completed_nodes = {n for n, s in cursor.node_states.items() if s == NodeState.SUCCESS}
        cursor.failed_nodes = {n for n, s in cursor.node_states.items() if s == NodeState.FAILED}
        cursor.skipped_nodes = {n for n, s in cursor.node_states.items() if s == NodeState.SKIPPED}

        return plan, cursor, registry
