"""
JSON Reporter.

Generates JSON reports from ExplanationGraphs.
"""
import json
import os
from typing import List

from crowdflow_dna.explainability.reporting.base import ReporterProtocol
from crowdflow_dna.explainability.reporting.registry import ReporterRegistry
from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.models import ExplanationGraph

@ReporterRegistry.register("json")
class JsonReporter(ReporterProtocol):
    
    def generate(self, session: ExplanationSession, graphs: List[ExplanationGraph]) -> None:
        report_path = os.path.join(session.output_directory, "explanation_report.json")
        
        report_data = {
            "session": {
                "session_id": session.session_id,
                "crowddna_version": session.crowddna_version,
                "schema_version": session.schema_version,
                "deployment_model_hash": session.deployment_model_hash,
                "deployment_artifact_hash": session.deployment_artifact_hash,
                "dataset_identifier": session.dataset_identifier,
                "enabled_explainers": session.enabled_explainers,
                "execution_timestamp": session.execution_timestamp,
                "execution_duration": session.execution_duration,
                "cache_status": session.cache_status
            },
            "explanations": []
        }
        
        for graph in graphs:
            exp_data = {
                "sequence_id": graph.sequence_id,
                "explainer_name": graph.explainer_name,
                "prediction_metadata": graph.prediction_metadata,
                "confidence_analysis": graph.confidence_analysis
            }
            # For brevity in JSON, we might summarize tensors or just store shapes,
            # or skip large tensors. We will just store max importance if available.
            if graph.node_importance is not None:
                exp_data["max_node_importance"] = graph.node_importance.max().item()
            report_data["explanations"].append(exp_data)
            
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=4)
