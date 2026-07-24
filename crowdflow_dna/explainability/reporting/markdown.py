"""
Markdown Reporter.

Generates Markdown reports from ExplanationGraphs.
"""
import os
from typing import List

from crowdflow_dna.explainability.reporting.base import ReporterProtocol
from crowdflow_dna.explainability.reporting.registry import ReporterRegistry
from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.models import ExplanationGraph

@ReporterRegistry.register("markdown")
class MarkdownReporter(ReporterProtocol):
    
    def generate(self, session: ExplanationSession, graphs: List[ExplanationGraph]) -> None:
        report_path = os.path.join(session.output_directory, "explanation_report.md")
        
        md_lines = [
            "# Explainability Report",
            "",
            "## Session Metadata",
            f"- **Session ID**: `{session.session_id}`",
            f"- **Deployment Model Hash**: `{session.deployment_model_hash}`",
            f"- **Dataset**: `{session.dataset_identifier}`",
            f"- **Timestamp**: `{session.execution_timestamp}`",
            f"- **Duration**: `{session.execution_duration:.2f}s`",
            f"- **Explainers Run**: `{', '.join(session.enabled_explainers)}`",
            "",
            "## Results Overview"
        ]
        
        for graph in graphs:
            md_lines.append(f"### Sequence: {graph.sequence_id} (Explainer: {graph.explainer_name})")
            if graph.prediction_metadata:
                md_lines.append("**Prediction Metadata**:")
                for k, v in graph.prediction_metadata.items():
                    md_lines.append(f"- {k}: {v}")
            if graph.confidence_analysis:
                md_lines.append("**Confidence Analysis**:")
                for k, v in graph.confidence_analysis.items():
                    md_lines.append(f"- {k}: {v:.4f}")
            if graph.node_importance is not None:
                md_lines.append(f"- **Max Node Importance**: {graph.node_importance.max().item():.4f}")
            md_lines.append("")
            
        with open(report_path, "w") as f:
            f.write("\n".join(md_lines))
