import os
import time
import logging
from typing import Dict, Optional
from training.orchestration.metadata import WorkflowManifest, NodeResult
from training.orchestration.artifacts import ArtifactRegistry
from training.orchestration.validation import WorkflowValidator
from training.orchestration.planner import WorkflowPlanner
from training.orchestration.recovery import RecoveryEngine
from training.orchestration.scheduler import WorkflowScheduler
from training.orchestration.executor import NodeExecutor
from training.orchestration.checkpoint import CheckpointManager
from training.orchestration.reporting.generator import ReportGenerator
from training.orchestration.reporting.models import ExecutionReport
from training.orchestration.registry import NodeRegistry
from training.orchestration.context import NodeContext
from training.orchestration.exceptions import NodeExecutionError, NodeCompensationError
from training.orchestration.state import NodeState

class WorkflowEngine:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def run(self, manifest: WorkflowManifest, resume_dir: Optional[str] = None) -> ExecutionReport:
        start_time = time.time()
        results: Dict[str, NodeResult] = {}
        
        # 1. Validation & Planning
        if resume_dir:
            self.logger.info(f"Resuming workflow from {resume_dir}")
            plan, cursor, artifact_registry = RecoveryEngine.resume(manifest, resume_dir)
            checkpoint_dir = resume_dir
        else:
            self.logger.info("Starting new workflow execution")
            WorkflowValidator.validate(manifest.definition)
            plan, cursor = WorkflowPlanner.plan(manifest.definition)
            artifact_registry = ArtifactRegistry()
            checkpoint_dir = os.path.join(manifest.execution.artifact_tree_root, "checkpoints")
            
        checkpoint_mgr = CheckpointManager(checkpoint_dir)
        
        from training.orchestration.dispatcher import EventDispatcher
        from training.orchestration.events import WorkflowEvent
        
        # Initial Checkpoint
        checkpoint_mgr.save(manifest.execution, plan, cursor, artifact_registry)
        
        EventDispatcher.dispatch(WorkflowEvent(
            workflow_id=manifest.definition.workflow_name,
            execution_id=manifest.execution.execution_id,
            correlation_id=manifest.execution.execution_id,
            payload={"action": "WORKFLOW_STARTED"}
        ))
        
        # 2. Execution Loop
        while True:
            ready_nodes = WorkflowScheduler.get_ready_nodes(plan, cursor)
            
            if not ready_nodes:
                # If no nodes are ready, and nothing is running, we are done.
                # In sequential model, this means graph is exhausted or deadlocked by failures.
                break
                
            for node_id in ready_nodes:
                node_meta = manifest.definition.nodes[node_id]
                node_cls = NodeRegistry.get(node_meta.node_type)
                
                context = NodeContext(
                    workflow_id=manifest.definition.workflow_name,
                    execution_id=manifest.execution.execution_id,
                    node_id=node_id,
                    correlation_id=f"{manifest.execution.execution_id}-{node_id}",
                    artifact_root=manifest.execution.artifact_tree_root,
                    artifact_registry=artifact_registry,
                    logger=self.logger
                )
                
                node_instance = node_cls()
                
                
                # Executor invocation
                try:
                    result = NodeExecutor.run(node_instance, context)
                except (NodeExecutionError, NodeCompensationError) as e:
                    self.logger.error(f"Node execution failed with orchestration error: {e}")
                    result = NodeResult(
                        node_id=node_id,
                        status=NodeState.FAILED,
                        error_message=str(e)
                    )
                    
                results[node_id] = result
                
                # Scheduler handles the result state transition
                WorkflowScheduler.handle_result(
                    plan, 
                    cursor, 
                    result, 
                    node_meta.failure_policy.isolation_level,
                    node_meta.retry_policy.max_retries
                )
                
                # Save checkpoint after every node transition
                checkpoint_mgr.save(manifest.execution, plan, cursor, artifact_registry)
                
                # Check execution policy abort condition
                if cursor.failed_nodes and manifest.definition.global_execution_policy.abort_on_any_failure:
                    self.logger.error("Workflow aborting due to global execution policy.")
                    break
                    
            if cursor.failed_nodes and manifest.definition.global_execution_policy.abort_on_any_failure:
                break

        # 3. Reporting
        report = ReportGenerator.generate(
            manifest=manifest,
            cursor=cursor,
            results=results,
            artifact_registry=artifact_registry,
            start_time=start_time
        )
        
        EventDispatcher.dispatch(WorkflowEvent(
            workflow_id=manifest.definition.workflow_name,
            execution_id=manifest.execution.execution_id,
            correlation_id=manifest.execution.execution_id,
            payload={"action": "WORKFLOW_COMPLETED", "status": report.status}
        ))
        
        return report
