import json
from dataclasses import asdict
from training.orchestration.reporting.models import ExecutionReport

class ReportExporter:
    @staticmethod
    def export_json(report: ExecutionReport, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(report), f, indent=2, sort_keys=True)
            
    @staticmethod
    def export_markdown(report: ExecutionReport, path: str) -> None:
        with open(path, 'w') as f:
            f.write(f"# Execution Report: {report.workflow_id}\n")
            f.write(f"Execution ID: {report.execution_id}\n")
            f.write(f"Status: **{report.status}**\n\n")
            
            f.write("## Performance\n")
            f.write(f"Total Duration: {report.performance.total_duration_seconds:.2f}s\n\n")
            
            f.write("## Nodes\n")
            for nr in report.node_reports:
                f.write(f"- **{nr.node_id}**: {nr.status} ({nr.duration_seconds:.2f}s)\n")
                
            if report.failure_reports:
                f.write("\n## Failures\n")
                for fr in report.failure_reports:
                    f.write(f"- **{fr.node_id}**: {fr.error_type} - {fr.error_message}\n")
                    
            f.write("\n## Artifacts\n")
            for ar in report.artifact_reports:
                f.write(f"- `{ar.artifact_id}` ({ar.schema_version}) -> {ar.location}\n")
