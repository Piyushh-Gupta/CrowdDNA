from typing import Dict
from training.orchestration.metadata import (
    WorkflowDefinition, WorkflowExecution, WorkflowManifest, NodeMetadata
)
from training.orchestration.policies import ExecutionPolicy
from training.framework.metadata import FrameworkMetadata

class PipelineIntegrator:
    """Helper to bridge the gap between Experiment Manager, CrowdFlowPipeline and Orchestration Engine."""
    
    @staticmethod
    def create_manifest(
        workflow_name: str, 
        execution_id: str, 
        session_id: str,
        artifact_root: str,
        nodes: Dict[str, NodeMetadata]
    ) -> WorkflowManifest:
        
        definition = WorkflowDefinition(
            workflow_name=workflow_name,
            workflow_schema_version="1.0.0",
            nodes=nodes,
            global_execution_policy=ExecutionPolicy()
        )
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            session_id=session_id,
            artifact_tree_root=artifact_root,
            framework_metadata=FrameworkMetadata(framework_version="1.0", crowddna_version="1.0")
        )
        
        return WorkflowManifest(definition=definition, execution=execution)
