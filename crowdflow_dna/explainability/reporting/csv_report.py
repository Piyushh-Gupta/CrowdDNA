"""
CSV Reporter.

Generates CSV reports from ExplanationGraphs.
"""
import csv
import os
from typing import List

from crowdflow_dna.explainability.reporting.base import ReporterProtocol
from crowdflow_dna.explainability.reporting.registry import ReporterRegistry
from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.models import ExplanationGraph

@ReporterRegistry.register("csv")
class CsvReporter(ReporterProtocol):
    
    def generate(self, session: ExplanationSession, graphs: List[ExplanationGraph]) -> None:
        report_path = os.path.join(session.output_directory, "explanation_report.csv")
        
        headers = [
            "sequence_id", 
            "explainer_name", 
            "predicted_class", 
            "base_probability", 
            "entropy", 
            "margin", 
            "max_node_importance"
        ]
        
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for graph in graphs:
                row = {
                    "sequence_id": graph.sequence_id,
                    "explainer_name": graph.explainer_name,
                    "predicted_class": graph.prediction_metadata.get("predicted_class", ""),
                    "base_probability": graph.prediction_metadata.get("base_probability", ""),
                    "entropy": "",
                    "margin": "",
                    "max_node_importance": ""
                }
                
                if graph.confidence_analysis:
                    row["entropy"] = f"{graph.confidence_analysis.get('entropy', 0.0):.4f}"
                    row["margin"] = f"{graph.confidence_analysis.get('margin', 0.0):.4f}"
                    
                if graph.node_importance is not None:
                    row["max_node_importance"] = f"{graph.node_importance.max().item():.4f}"
                    
                writer.writerow(row)
