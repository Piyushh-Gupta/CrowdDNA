from typing import Set
from training.orchestration.metadata import WorkflowDefinition
from training.orchestration.exceptions import WorkflowValidationError, WorkflowCycleError

class WorkflowValidator:
    @staticmethod
    def validate(definition: WorkflowDefinition) -> None:
        WorkflowValidator._validate_schema(definition)
        WorkflowValidator._validate_node_ids(definition)
        WorkflowValidator._validate_dependencies(definition)
        WorkflowValidator._validate_cycles(definition)
        WorkflowValidator._validate_artifact_destinations(definition)
        
    @staticmethod
    def _validate_schema(definition: WorkflowDefinition) -> None:
        if not definition.workflow_schema_version:
            raise WorkflowValidationError("workflow_schema_version is required.")
            
    @staticmethod
    def _validate_node_ids(definition: WorkflowDefinition) -> None:
        seen = set()
        for node_id in definition.nodes:
            if node_id in seen:
                raise WorkflowValidationError(f"Duplicate node ID found: {node_id}")
            seen.add(node_id)
            
    @staticmethod
    def _validate_dependencies(definition: WorkflowDefinition) -> None:
        all_nodes = set(definition.nodes.keys())
        for node_id, meta in definition.nodes.items():
            for dep in meta.dependencies:
                if dep not in all_nodes:
                    raise WorkflowValidationError(f"Node '{node_id}' depends on missing node '{dep}'")
                    
    @staticmethod
    def _validate_cycles(definition: WorkflowDefinition) -> None:
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in definition.nodes[node_id].dependencies:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
                    
            rec_stack.remove(node_id)
            return False
            
        for node_id in definition.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    raise WorkflowCycleError(f"Cycle detected involving node '{node_id}'")
                    
    @staticmethod
    def _validate_artifact_destinations(definition: WorkflowDefinition) -> None:
        # Prevent trivial collision where multiple nodes declare exact same static artifact names without isolation.
        # Currently, context isolates by artifact_root, but we can do a sanity check on configurations.
        pass
