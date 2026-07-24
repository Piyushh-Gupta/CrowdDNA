import time
from typing import List, Dict
from training.orchestration.metadata import WorkflowManifest, ExecutionCursor, NodeResult
from training.orchestration.artifacts import ArtifactRegistry
from training.orchestration.reporting.models import ExecutionReport, NodeReport, ArtifactReport, FailureReport, PerformanceReport

class ReportGenerator:
    @staticmethod
    def generate(manifest: WorkflowManifest, cursor: ExecutionCursor, results: Dict[str, NodeResult], artifact_registry: ArtifactRegistry, start_time: float) -> ExecutionReport:
        total_duration = time.time() - start_time
        
        node_reports: List[NodeReport] = []
        failure_reports: List[FailureReport] = []
        
        for node_id in manifest.definition.nodes.keys():
            state = cursor.get_state(node_id)
            result = results.get(node_id)
            retries = cursor.retry_counts.get(node_id, 0)
            
            if result:
                node_reports.append(NodeReport(
                    node_id=node_id,
                    status=state.name,
                    duration_seconds=result.metrics.get("duration_seconds", 0.0),
                    retries=retries,
                    artifacts_produced=list(result.artifacts_produced),
                    metrics=dict(result.metrics)
                ))
                if result.error_message:
                    failure_reports.append(FailureReport(
                        node_id=node_id,
                        error_type="ExecutionError",
                        error_message=result.error_message
                    ))
            else:
                node_reports.append(NodeReport(
                    node_id=node_id,
                    status=state.name,
                    duration_seconds=0.0,
                    retries=retries
                ))

        artifact_reports: List[ArtifactReport] = [
            ArtifactReport(
                artifact_id=rec.artifact_id,
                producer_node_id=rec.producer_node_id,
                schema_version=rec.schema_version,
                checksum=rec.checksum,
                location=rec.location
            ) for artifact_id, rec in sorted(artifact_registry.get_all().items())
        ]
        
        perf_report = PerformanceReport(total_duration_seconds=total_duration)
        
        # Determine global status
        if cursor.failed_nodes and manifest.definition.global_execution_policy.abort_on_any_failure:
            global_status = "FAILED"
        elif cursor.failed_nodes:
            global_status = "PARTIAL_SUCCESS"
        else:
            global_status = "SUCCESS"

        return ExecutionReport(
            workflow_id=manifest.definition.workflow_name,
            execution_id=manifest.execution.execution_id,
            status=global_status,
            node_reports=node_reports,
            artifact_reports=artifact_reports,
            failure_reports=failure_reports,
            performance=perf_report
        )
